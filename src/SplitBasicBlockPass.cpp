#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct SplitBasicBlockPass : public FunctionPass {
  static char ID;
  SplitBasicBlockPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    for (auto &B : F) {
      // 注意：这只是一个示例，真实场景下可能不会这样简单地拆分每个基本块
      if (!B.empty()) {
        // SplitBlock函数需要一个要拆分的基本块、拆分点指令和一个Pass的上下文
        BasicBlock *BB = &B;
        Instruction *SplitPt = &(B.front());
        SplitBlock(BB, SplitPt);
        return true; // 表示基本块被修改
      }
    }
    return false; // 没有修改任何基本块
  }
}; // 结构体结束

// LLVM的Pass注册机制
char SplitBasicBlockPass::ID = 0;
static RegisterPass<SplitBasicBlockPass> X("splitbb", "Split Basic Block Pass", false, false);
} // 匿名命名空间结束
