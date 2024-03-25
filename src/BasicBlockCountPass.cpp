#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include <fstream>
#include <iostream>
#include <string>
#include "OutputHandler.hpp"

using namespace llvm;
using namespace std;

static cl::opt<std::string> OutputPrefix("output-prefix", cl::desc("Specify prefix for output files"), cl::value_desc("prefix"));

// 定义LLVM pass
class BasicBlockCountPass : public FunctionPass {
public:
    static char ID;
    OutputHandler handler;
    BasicBlockCountPass() : FunctionPass(ID),handler(OutputPrefix.getValue()) {
        
    }

    bool runOnFunction(Function &F) override {
        
        unsigned int BBCounter = 0;  // Count the basic blocks
        for (auto &B : F) {
            unsigned int InstCounter = 0; // Count the instructions in each basic block

            for (auto &I : B) {
                InstCounter++;
            }

            handler.writeBasicBlockInfo(F, BBCounter, InstCounter);
            BBCounter++;
        }

        return false; // Not modifying the IR
    }
};

char BasicBlockCountPass::ID = 0;

// 注册pass
static RegisterPass<BasicBlockCountPass> X("BasicBlockCountPass", "BasicBlock Count", false, false);
