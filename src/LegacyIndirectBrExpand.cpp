//===- IndirectBrExpandPass.cpp - Expand indirectbr to switch -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Implements an expansion pass to turn `indirectbr` instructions in the IR
/// into `switch` instructions. This works by enumerating the basic blocks in
/// a dense range of integers, replacing each `blockaddr` constant with the
/// corresponding integer constant, and then building a switch that maps from
/// the integers to the actual blocks. All of the indirectbr instructions in the
/// function are redirected to this common switch.
///
/// While this is generically useful if a target is unable to codegen
/// `indirectbr` natively, it is primarily useful when there is some desire to
/// get the builtin non-jump-table lowering of a switch even when the input
/// source contained an explicit indirect branch construct.
///
/// Note that it doesn't make any sense to enable this pass unless a target also
/// disables jump-table lowering of switches. Doing that is likely to pessimize
/// the code.
///
//===----------------------------------------------------------------------===//

#include "LegacyIndirectBrExpand.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "indirectbr-expand"

namespace {

class IndirectBrExpandPass : public FunctionPass {

public:
    static char ID; // Pass identification, replacement for typeid

    IndirectBrExpandPass() : FunctionPass(ID) {
        initializeIndirectBrExpandPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override;
};

} // end anonymous namespace

char IndirectBrExpandPass::ID = 0;

// INITIALIZE_PASS(IndirectBrExpandPass, DEBUG_TYPE,
//                 "Expand indirectbr instructions", false, false)

FunctionPass *llvm::createLegacyIndirectBrExpandPass() {
    return new IndirectBrExpandPass();
}

// 这段代码定义了一个LLVM的函数遍历，目的是扩展（替换）函数中的所有间接分支指令（indirectbr）为更
// 具体的、基于swtich指令的结构。这种转换可以提高在某些情况下的代码性能，尤其是在能够确定间接跳转可能的
// 目标时。
bool IndirectBrExpandPass::runOnFunction(Function &F) {
    //获取数据布局信息，这是与目标平台相关的信息，如类型大小和对齐要求
    auto &DL = F.getParent()->getDataLayout();
    // 定义一个向量，用于存储要重写的indirectbr指令
    SmallVector<IndirectBrInst *, 1> IndirectBrs;

    // 定义一个集合，用于存储所有可能的indirectbr指令的后续基本块
    // Set of all potential successors for indirectbr instructions.
    SmallPtrSet<BasicBlock *, 4> IndirectBrSuccs;

    // Build a list of indirectbrs that we want to rewrite.
    // 遍历函数中的所有基本块，寻找那些以indirectbr作为终结指令的基本块，并处理每个找到的indirectbr指令
    for (BasicBlock &BB : F)
        if (auto *IBr = dyn_cast<IndirectBrInst>(BB.getTerminator())) {
            // Handle the degenerate case of no successors by replacing the indirectbr
            // with unreachable as there is no successor available.
            // 如果indirectbr没有任何后继，就用一个不可达指令unreachable替换它，因为这意味着，程序在运行时不可能正确地执行到这里
            if (IBr->getNumSuccessors() == 0) {
                (void)new UnreachableInst(F.getContext(), IBr);
                IBr->eraseFromParent();
                continue;
            }
            // 将indirectbr指令添加到向量中，并将其所有可能的后继基本块添加到集合中
            IndirectBrs.push_back(IBr);
            for (BasicBlock *SuccBB : IBr->successors())
                IndirectBrSuccs.insert(SuccBB);
        }

    if (IndirectBrs.empty())
        return false;

    // If we need to replace any indirectbrs we need to establish integer
    // constants that will correspond to each of the basic blocks in the function
    // whose address escapes. We do that here and rewrite all the blockaddress
    // constants to just be those integer constants cast to a pointer type.
    // 定义一个向量存储那些地质被取代的基本块
    SmallVector<BasicBlock *, 4> BBs;

    for (BasicBlock &BB : F) {
        // Skip blocks that aren't successors to an indirectbr we're going to
        // rewrite.
        // 如果当前基本块不是任何‘indirectbr’指令的潜在后继，则跳过它
        
        if (!IndirectBrSuccs.count(&BB))
            continue;
        
        //定义一个lambda函数，用于检查一个使用use是否是块地址（BlockAddress）的使用
        auto IsBlockAddressUse = [&](const Use &U) {
            return isa<BlockAddress>(U.getUser());
        };
        //在基本块的使用中寻找块地址的使用，如果没有找到，则跳过当前基本块。
        auto BlockAddressUseIt = llvm::find_if(BB.uses(), IsBlockAddressUse);
        if (BlockAddressUseIt == BB.use_end())
            continue;
        
        assert(std::find_if(std::next(BlockAddressUseIt), BB.use_end(),
                            IsBlockAddressUse) == BB.use_end() &&
               "There should only ever be a single blockaddress use because it is "
               "a constant and should be uniqued.");
        //获取到块地址（BlockAddress）的实际对象。
        auto *BA = cast<BlockAddress>(BlockAddressUseIt->getUser());

        // Skip if the constant was formed but ended up not being used (due to DCE
        // or whatever).
        if (!BA->isConstantUsed())
            continue;

        // Compute the index we want to use for this basic block. We can't use zero
        // because null can be compared with block addresses.
        // 为当前基本块分配一个唯一的索引，并将其添加到BBs向量中。
        int BBIndex = BBs.size() + 1;
        BBs.push_back(&BB);
        //创建一个整数常量，其值为当前基本块的索引，类型为适合存储指针的整数类型。
        auto *ITy = cast<IntegerType>(DL.getIntPtrType(BA->getType()));
        ConstantInt *BBIndexC = ConstantInt::get(ITy, BBIndex);

        // Now rewrite the blockaddress to an integer constant based on the index.
        // FIXME: This part doesn't properly recognize other uses of blockaddress
        // expressions, for instance, where they are used to pass labels to
        // asm-goto. This part of the pass needs a rework.
        //将块地址的所有使用替换为一个整数到指针的常量表达式，这个表达式的整数值是上面分配的索引。
        BA->replaceAllUsesWith(ConstantExpr::getIntToPtr(BBIndexC, BA->getType()));
    }

    //处理没有块地址使用的indirectbr指令
    if (BBs.empty()) {
        //如果没有基本块的地址被取，则所有indirectbr指令都不能获得有效输入，将它们替换为unreachable指令。
        // There are no blocks whose address is taken, so any indirectbr instruction
        // cannot get a valid input and we can replace all of them with unreachable.
        for (auto *IBr : IndirectBrs) {
            (void)new UnreachableInst(F.getContext(), IBr);
            IBr->eraseFromParent();
        }
        return true;
    }

    // 创建和配置switch指令
    // 声明变量以存储新switch指令所在的基本块和switch指令的比较值。
    BasicBlock *SwitchBB;
    Value *SwitchValue;

    // Compute a common integer type across all the indirectbr instructions.
    // 计算所有indirectbr指令中地址类型的公共整数类型。
    IntegerType *CommonITy = nullptr;
    for (auto *IBr : IndirectBrs) {
        auto *ITy =
            cast<IntegerType>(DL.getIntPtrType(IBr->getAddress()->getType()));
        if (!CommonITy || ITy->getBitWidth() > CommonITy->getBitWidth())
            CommonITy = ITy;
    }
    //定义一个lambda函数，用于为每个indirectbr指令创建一个将地址转换为公共整数类型的指针转换指令。
    auto GetSwitchValue = [DL, CommonITy](IndirectBrInst *IBr) {
        return CastInst::CreatePointerCast(
            IBr->getAddress(), CommonITy,
            Twine(IBr->getAddress()->getName()) + ".switch_cast", IBr);
    };
    //如果只有一个indirectbr指令，直接在其所在基本块中进行替换。
    //如果有多个，则创建一个新的基本块来放置switch指令，并更新所有indirectbr指令跳转到这个新基本块。
    if (IndirectBrs.size() == 1) {
        // If we only have one indirectbr, we can just directly replace it within
        // its block.
        SwitchBB = IndirectBrs[0]->getParent();
        SwitchValue = GetSwitchValue(IndirectBrs[0]);
        IndirectBrs[0]->eraseFromParent();
    } else {
        // Otherwise we need to create a new block to hold the switch across BBs,
        // jump to that block instead of each indirectbr, and phi together the
        // values for the switch.
        SwitchBB = BasicBlock::Create(F.getContext(), "switch_bb", &F);
        auto *SwitchPN = PHINode::Create(CommonITy, IndirectBrs.size(),
                                         "switch_value_phi", SwitchBB);
        SwitchValue = SwitchPN;

        // Now replace the indirectbr instructions with direct branches to the
        // switch block and fill out the PHI operands.
        for (auto *IBr : IndirectBrs) {
            SwitchPN->addIncoming(GetSwitchValue(IBr), IBr->getParent());
            BranchInst::Create(SwitchBB, IBr);
            IBr->eraseFromParent();
        }
    }

    // Now build the switch in the block. The block will have no terminator
    // already.
    // 在新的基本块中创建一个switch指令
    auto *SI = SwitchInst::Create(SwitchValue, BBs[0], BBs.size(), SwitchBB);

    // Add a case for each block.
    // 为switch指令添加case，每个case对应一个可能的跳转目标基本块
    for (int i : llvm::seq<int>(1, BBs.size()))
        SI->addCase(ConstantInt::get(CommonITy, i + 1), BBs[i]);
    /**
     * 
     * indirectbr <type> <value>, [label <dest1>, label <dest2>, ...]
     * 
    */
    return true;
}