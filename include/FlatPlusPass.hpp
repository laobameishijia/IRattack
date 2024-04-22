#pragma once
#include "llvm/Pass.h"

namespace llvm {
// Pass *createFlatPlus(bool flag, bool dont_fla_invoke=false, int fla_cnt=1);
FunctionPass *createFlatPlus(bool flag, bool dont_fla_invoke=false, int fla_cnt=1);
} // namespace llvm
