#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InlineAsm.h" // Include InlineAsm header
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

using namespace llvm;
using namespace std;

struct BlockInfo {
    int count;  // 基本块个数
    std::vector<int> asmIndices;  // 汇编指令序数列表
};

struct FunctionInfo {
    std::unordered_map<int, BlockInfo> blocks;  // 使用基本块序数作为键
};

// 解析字符串中的所有+号后面的数字，并添加到向量中
void parseAsmIndices(const std::string& str, std::vector<int>& indices) {
    std::istringstream stream(str);
    char ch;
    int num;
    while (stream >> ch >> num) {
        if (ch == '+') {
            indices.push_back(num);
        }
    }
}

// 解析文件并填充包含多个函数信息的map
void parseFile(const std::string& filePath, std::unordered_map<std::string, FunctionInfo>& functionsmap) {
    std::ifstream file(filePath);
    std::string line;

    while (getline(file, line)) {
        std::istringstream iss(line);
        std::string functionName, blockPart, asmPart;
        getline(iss, functionName, '#');

        getline(iss, blockPart, '&');
        int blockNum = stoi(blockPart);  // 基本块序数

        getline(iss, asmPart, ':');
        int count = stoi(asmPart);  // 基本块个数

        std::vector<int> asmIndices;
        if (getline(iss, asmPart)) {
            parseAsmIndices(asmPart, asmIndices);  // 解析汇编指令序数
        }

        // 存储解析结果
        functionsmap[functionName].blocks[blockNum] = {count, asmIndices};
    }
}

static cl::opt<string> Basicblockfilepath("Basicblockfilepath", cl::desc("the path of Basicblockfile"), cl::value_desc("string"));

class CustomAsmInsertionPass : public FunctionPass
{
public:
  static char ID;
  std::unordered_map<std::string, FunctionInfo> functionsmap;

  std::vector<std::string> asm_instruction_array = {
    "nop\n",
    "subq 0, %rax\n",
    "addq 0, %rax\n",
    "leaq (%rax), %rax\n",
    "movq %rax, %rax\n",
    "xchgq %rax, %rax\n",
    "pushfq\n pushq %rax\n xorl %eax, %eax\n cmovol %ecx, %eax\n popq %rax\n popfq\n", // OF溢出标志
    "pushfq\n pushq %rax\n xorl %eax, %eax\n cmovpl %eax, %eax\n popq %rax\n popfq\n", // PF奇偶校验标志
    "pushfq\n cmpq %rax, %rax\n cmovb %eax, %eax\n popfq\n",                           // CF进位标志、ZF零标志 条件移动指令
    "pushfq\n cmpq %rax, %rax\n cmovg %ecx, %eax\n popfq\n",                           // ZF零标志、SF符号标志、溢出标志OF
    "pushfq\n cmpq %rax, %rax\n cmovs %ecx, %eax\n popfq\n",                           // SF符号标志
    "pushfq\n cmpq %rax, %rax\n cmovl %ecx, %eax\n popfq\n",                           // 小于的条件下移动数据
    "pushfq\n cmpq %rax, %rax\n cmovns %eax, %eax\n popfq\n",                          // 在符号标志未设置时移动数据
    "pushfq\n pushq %rax\n xorl %eax, %eax\n cmovnp %ecx, %eax\n popq %rax\n popfq\n", // 在奇偶校验标志未设置时移动数据
    "pushfq\n cmpq %rax, %rax\n cmovno %ecx, %eax\n popfq\n",                          // 溢出标志被设置时移动数据
    "addq 1, %rax\n subq 1, %rax\n",
    "subq -2, %rax\n addq 2, %rax\n",
    "pushq %rax\n negq %rax\n negq %rax\n popq %rax\n", // 求补码操作
    "notq %rax\n notq %rax\n",                          // 取反操作
    "pushq %rax\n popq %rax\n",
    "pushfq\n popfq\n", // 保存标志寄存器 f-16位 fd-32位 fq-64位
    "xchgq %rax, %rcx\n xchgq %rcx, %rax\n",
    "pushq %rax\n notq %rax\n popq %rax\n",
    "xorq %rbx, %rax\n xorq %rax, %rbx\n xorq %rax, %rbx\n xorq %rbx, %rax\n",
    "popq %rbx\n movq %rax, %rbx\n addq 1, %rax\n movq %rbx, %rax\n popq %rbx\n",
    "pushq %rax\n incq %rax\n decq %rax\n decq %rax\n popq %rax\n",
    "pushq %rbx\n movq %rax, %rbx\n cmpq %rax, %rax\n setg %al\n movzbq %al, %rax\n movq %rbx, %rax\n popq %rbx\n", // setg指令根据比较结果设置条件标志
};

  CustomAsmInsertionPass() : FunctionPass(ID) {
    string basicblockpath = Basicblockfilepath.getValue();
    parseFile(basicblockpath, functionsmap);
    // 示例输出
    // 文件解析输出没有问题
    // for (const auto& func : functionsmap) {
    //     std::cout << "Function: " << func.first << std::endl;
    //     for (const auto& block : func.second.blocks) {
    //         std::cout << "  Block #" << block.first << ", Count: " << block.second.count << ", Asm Indices: ";
    //         for (int index : block.second.asmIndices) {
    //             std::cout << index << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }
  }

  bool runOnFunction(Function &F) override {
    std::string functionName = F.getName().str();
    // 检查这个函数是否在我们的映射中
    // std::cout <<  "进入插件\n";
    if (functionsmap.find(functionName) != functionsmap.end()) {
      // std::cout <<  "找到函数\n" << functionName << std::endl;
      unsigned int BBCounter = 0;
      for (BasicBlock &BB : F) {
        // 检查每个基本块是否应该插入汇编指令
        if (functionsmap[functionName].blocks.find(BBCounter) != functionsmap[functionName].blocks.end()) {
          // std::cout <<  "找到基本块\n" << BBCounter << std::endl;
          for (int asmIndex : functionsmap[functionName].blocks[BBCounter].asmIndices) {
            // 插入汇编指令
            if (!BB.empty()) {
              Instruction *firstInst = &BB.front();
              LLVMContext &Ctx = F.getContext();
              InlineAsm *customAsm = InlineAsm::get(
                FunctionType::get(Type::getVoidTy(Ctx), false),
                asm_instruction_array[asmIndex], // 使用 asmIndex 选择指令
                "",
                true, // HasSideEffects
                false // IsAlignStack
              );
              CallInst::Create(customAsm, "", firstInst);
            }
            // std::cout <<  "成功插入\n" << std::endl;
          }
        }
        BBCounter++;
      }
    }
    return true;
  }
};


char CustomAsmInsertionPass::ID = 0;
static RegisterPass<CustomAsmInsertionPass> X("fuzz-asm-insertion", "Insert custom assembly instruction at the beginning of each basic block", false, false);




// int main() {
//     std::unordered_map<std::string, FunctionInfo> functions;
//     parseFile("your_file_path.txt", functions);

//     // 示例输出
//     for (const auto& func : functions) {
//         std::cout << "Function: " << func.first << std::endl;
//         for (const auto& block : func.second.blocks) {
//             std::cout << "  Block #" << block.first << ", Count: " << block.second.count << ", Asm Indices: ";
//             for (int index : block.second.asmIndices) {
//                 std::cout << index << " ";
//             }
//             std::cout << std::endl;
//         }
//     }

//     return 0;
// }

//   bool runOnFunction(Function &F) override
//   {
//     string functionnamestr = IRfunctionName.getValue();
//     int asminstructionID = AsminstructionID.getValue();
//     size_t pos = functionnamestr.find("#");

//     string functionname = functionnamestr.substr(0, pos);
//     string BBnumber_str = functionnamestr.substr(pos + 1);
//     int BBnumber = stoi(BBnumber_str);
//     // std::cout << "函数名为: " << functionname << "\n";
//     // std::cout << "基本块序号为: " << BBnumber_str << "\n";

//     unsigned int BBCounter = 0; // Count the basic blocks
//     for (BasicBlock &BB : F)
//     {
//       if (F.getName().str() == functionname && BBCounter == BBnumber)
//       {

//         // Insert custom assembly instruction at the beginning of each basic block
//         if (!BB.empty())
//         {
//           Instruction *firstInst = &BB.front();
//           Module *M = F.getParent();
//           LLVMContext &Ctx = M->getContext();

//           InlineAsm *customAsm = InlineAsm::get(
//               FunctionType::get(Type::getVoidTy(Ctx), false),
//               // "nop", //这个地方可以修改为别的东西
//               asm_instruction_array[asminstructionID],
//               "",
//               true, // HasSideEffects
//               false // IsAlignStack
//           );

//           CallInst::Create(customAsm, "", firstInst);
//           // std::cout << "成功插入 " << "\n";
//         }
//       }
//       BBCounter ++;
//     }
//     return true; // Function has been modified
//   }


