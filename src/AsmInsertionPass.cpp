#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InlineAsm.h"  // Include InlineAsm header
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <string>

using namespace llvm;
using namespace std;

static cl::opt<string> IRfunctionName("IRfunctionName", cl::desc("Specify id of asminstructions"), cl::value_desc("string"));
static cl::opt<int> AsminstructionID("asminstructionID", cl::desc("Specify id of asminstructions"), cl::value_desc("int"));

class CustomAsmInsertionPass : public FunctionPass {
public:
  static char ID;
  vector<string> asm_instruction_array = {
    "nop\n", 
    "sub eax,0\n",
    "add eax,0\n",
    "lea eax,[eax+0]\n",
    "mov eax,eax\n",
    "xchg eax,eax\n",
    "pushfd\n push eax\n xor eax, eax\n comvo eax,ecx\n pop eax\n popfd\n", //OF溢出标志
    "pushfd\n push eax\n xor eax, eax\n comvp eax,eax\n pop eax\n popfd\n", //PF奇偶校验标志
    "pushfd\n cmp eax, eax\n comva eax, eax\n popfd\n",//CF进位标志、ZF零标志 条件移动指令
    "pushfd\n cmp eax, eax\n comvg eax, ecx\n popfd\n",//ZF零标志、SF符号标志、溢出标志OF  
    "pushfd\n push eax\n mov eax, -1\n cmp eax, 0\n cmovs eax, eax\n pop eax\n pop fd\n",//SF符号标志
    "pushfd\n cmp eax, eax\n cmovl eax, ecx\n popfd\n",//小于的条件下移动数据
    "pushfd\n cmp eax, eax\n cmovns eax, eax\n popfd\n",//在符号标志未设置时移动数据
    "pushfd\n push eax\n xor eax,eax\n cmovnp eax, ecx\n pop eax\n popfd\n",//在奇偶校验标志未设置时移动数据
    "pushfd\n cmp eax, eax\n cmovno eax,ecx\n popfd\n",//溢出标志被设置时移动数据
    "add eax, 1\n sub eax,1\n",
    "sub eax, -2\n add eax, 2\n",
    "push eax\n neg eax\n neg eax\n pop eax\n",//求补码操作
    "NOT eax\n NOT eax\n",//取反操作
    "push eax\n pop eax\n",
    "pushfd\n popfd\n",//保存标志寄存器 f-16位 fd-32位 fq-64位
    "xchg eax, ecx\n xchg ecx,eax\n",
    "push eax\n not eax\n pop eax\n",
    "xor eax, ebx\n xor ebx, eax\n xor ebx, eax\n xor eax, ebx\n",
    "pop ebx\n mov ebx, eax\n add eax,1\n mov eax, ebx\n pop ebx\n",
    "push eax\n inc eax\n dec eax\n dec eax\n pop eax\n",
    "push ebx\n mov ebx, eax\n cmp eax, eax\n setg al\n movzx eax, al\n mov eax, ebx\n pop ebx\n",//setg指令根据比较结果设置条件标志
};
  CustomAsmInsertionPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    string functionnamestr = IRfunctionName.getValue();
    int asminstructionID = AsminstructionID.getValue();
    size_t pos = functionnamestr.find("#");

    string functionname = functionnamestr.substr(0, pos);
    string BBnumber_str = functionnamestr.substr(pos + 1);
    int BBnumber = stoi(BBnumber_str);

    unsigned int BBCounter = 0;  // Count the basic blocks
    for (BasicBlock &BB : F) {
      if (F.getName().str() == functionname && BBCounter == BBnumber ){

        // Insert custom assembly instruction at the beginning of each basic block
        if (!BB.empty()) {
          Instruction *firstInst = &BB.front();
          Module *M = F.getParent();
          LLVMContext &Ctx = M->getContext();

          InlineAsm *customAsm = InlineAsm::get(
              FunctionType::get(Type::getVoidTy(Ctx), false),
              // "nop", //这个地方可以修改为别的东西
              asm_instruction_array[asminstructionID],
              "",
              true,  // HasSideEffects
              false  // IsAlignStack
          );

          CallInst::Create(customAsm, "", firstInst);
        }
      }
    }
    return true; // Function has been modified
  }
};

char CustomAsmInsertionPass::ID = 0;
static RegisterPass<CustomAsmInsertionPass> X("custom-asm-insertion", "Insert custom assembly instruction at the beginning of each basic block", false, false);
