#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <random>
#include <set>
#include <map>
#include <vector>

using namespace llvm;

struct labelInfo {
    uint32_t x;
    uint32_t y;
    uint32_t label;
};

class FlatPlus {
public:
    FlatPlus();

    bool doFlat(Function &F);

protected:
    std::mt19937 rng;
    uint32_t subTransCnt = 0;
    uint32_t imm32 = 0;
    uint8_t *imm8 = nullptr;
    AllocaInst *bakPtr[2] = {nullptr};

    std::map<BasicBlock *, struct labelInfo> blockInfos;
    std::set<uint32_t> labelSet;

    // init random parameter
    void initRandom();

    // detail translate algo
    uint32_t genLabel(uint32_t x);

    void initBlockInfo();
    void genBlockInfo(BasicBlock *bb);

    void allocTransBlockPtr(IRBuilder<> &builder);

    BasicBlock **genTransBlocks(
        Function &F, Value *xPtr, Value *yPtr, Value *labelPtr);

    void shuffleBlock(SmallVector<BasicBlock *, 0> &bb);
};

class FlatPlusPass : public FunctionPass {
public:
    static char ID;
    bool flag;
    bool DontFlaInvoke;
    int FlaCnt;

    FlatPlusPass(bool dont_fla_invoke=false, int fla_cnt=1) : FunctionPass(ID) {
        flatPlus = new FlatPlus();
        flag = true;
        DontFlaInvoke = dont_fla_invoke;
        FlaCnt = fla_cnt;
    }
    FlatPlusPass(bool flag, bool dont_fla_invoke=false, int fla_cnt=1) : FunctionPass(ID) {
        flatPlus = new FlatPlus();
        this->flag = flag;
        DontFlaInvoke = dont_fla_invoke;
        FlaCnt = fla_cnt;
    }

    bool runOnFunction(Function &F) override;

private:
    FlatPlus *flatPlus = nullptr;
};
