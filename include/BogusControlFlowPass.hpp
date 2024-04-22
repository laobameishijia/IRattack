#pragma once
#include "llvm/Pass.h"

namespace llvm {
// Pass *createBogusControlFlow(bool flag, int bcf_rate=30);
FunctionPass *createBogusControlFlow(bool flag, int bcf_rate=30);
} // namespace llvm
