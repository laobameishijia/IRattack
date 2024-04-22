#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <random>
#include <set>
#include <vector>

using namespace llvm;

typedef uint32_t primeTy;

class BogusControlFlow {
public:
    BogusControlFlow(int bcf_rate);

    bool doBogusControlFlow(Function &F);

protected:
    std::mt19937 rng;
    std::vector<Value *> usableVars;
    bool firstObf;
    int BCFRate;

    void collectUsableVars(std::vector<BasicBlock *> &useful);
    void buildBCF(BasicBlock *src, BasicBlock *dst,
                  std::vector<BasicBlock *> &jumpTarget,
                  Function &F);
    BasicBlock * buildJunk(Function &F);
};

class BogusControlFlowPass : public FunctionPass {
public:
    static char ID ;
    bool flag;

    BogusControlFlowPass(int bcf_rate) : FunctionPass(ID) {
        bogusControlFlow = new BogusControlFlow(bcf_rate);
        flag = true;
    }
    BogusControlFlowPass(bool flag, int bcf_rate) : FunctionPass(ID) {
        bogusControlFlow = new BogusControlFlow(bcf_rate);
        this->flag = flag;
    }

    bool runOnFunction(Function &F) override;

private:
    BogusControlFlow *bogusControlFlow = nullptr;
};
