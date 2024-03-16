#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InlineAsm.h"  // Include InlineAsm header
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class CustomAsmInsertionPass : public FunctionPass {
public:
  static char ID;
  CustomAsmInsertionPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    for (BasicBlock &BB : F) {
      // Insert custom assembly instruction at the beginning of each basic block
      if (!BB.empty()) {
        Instruction *firstInst = &BB.front();
        Module *M = F.getParent();
        LLVMContext &Ctx = M->getContext();

        InlineAsm *customAsm = InlineAsm::get(
            FunctionType::get(Type::getVoidTy(Ctx), false),
            "nop", //这个地方可以修改为别的东西
            "",
            true,  // HasSideEffects
            false  // IsAlignStack
        );

        CallInst::Create(customAsm, "", firstInst);
      }
    }
    return true; // Function has been modified
  }
};
} // end anonymous namespace

char CustomAsmInsertionPass::ID = 0;
static RegisterPass<CustomAsmInsertionPass> X("custom-asm-insertion", "Insert custom assembly instruction at the beginning of each basic block", false, false);
