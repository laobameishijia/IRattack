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

#include "BogusControlFlowPass.hpp"
#include "FlatPlusPass.hpp"

using namespace llvm;
using namespace std;

struct BlockMutationInfo {
    std::vector<int> asmInstructionsIndices;
};

struct FunctionMutationInfo {
    int flattenLevel;
    int bcfRate;
    std::unordered_map<int, BlockMutationInfo> blockInfos;  // key: block index
};

using MutationInfo = std::unordered_map<std::string, FunctionMutationInfo>;  // key: function name

MutationInfo parseMutationFile(const std::string& filename) {
    MutationInfo mutations;
    std::ifstream file(filename);
    std::string line;

    std::string currentFunction;
    int currentFlattenLevel = 0, currentBCFRate = 0;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == ';') continue;  // Skip comments and empty lines
        
        if (line[0] == '[') {  // New function settings
            auto endPos = line.find(']');
            auto atPos = line.find('@');
            auto commaPos = line.find(',', atPos);
            currentFunction = line.substr(1, atPos - 1);
            currentFlattenLevel = std::stoi(line.substr(atPos + 1, commaPos - atPos - 1));
            currentBCFRate = std::stoi(line.substr(commaPos + 1, endPos - commaPos - 1));
            continue;
        }

        auto hashPos = line.find('#');
        auto andPos = line.find('&');
        auto colonPos = line.find(':');

        int blockIndex = std::stoi(line.substr(hashPos + 1, andPos - hashPos - 1));
        
        BlockMutationInfo blockInfo;

        if (colonPos != std::string::npos) {  // There are assembly instructions to insert
            std::stringstream ss(line.substr(colonPos + 1));  // Skip ":"
            std::string token;
            while (std::getline(ss, token, '+')) {
                if (!token.empty()) {
                    blockInfo.asmInstructionsIndices.push_back(std::stoi(token));
                }
            }
        }

        mutations[currentFunction].flattenLevel = currentFlattenLevel;
        mutations[currentFunction].bcfRate = currentBCFRate;
        mutations[currentFunction].blockInfos[blockIndex] = std::move(blockInfo);
    }

    return mutations;
}

void printModuleMutationInfo(const MutationInfo& moduleInfo) {
    for (const auto& functionPair : moduleInfo) {
        const auto& functionName = functionPair.first;
        const auto& functionInfo = functionPair.second;

        // 输出函数级别的变异配置
        std::cout << "[" << functionName << "@" << functionInfo.flattenLevel 
                  << "," << functionInfo.bcfRate << "]" << std::endl;

        for (const auto& blockPair : functionInfo.blockInfos) {
            const auto& blockIndex = blockPair.first;
            const auto& blockInfo = blockPair.second;

            // 输出基本块级别的变异信息
            std::cout << functionName << "#" << blockIndex << "&" 
                      << blockInfo.asmInstructionsIndices.size() << ":";

            for (size_t i = 0; i < blockInfo.asmInstructionsIndices.size(); ++i) {
                std::cout << "+" << blockInfo.asmInstructionsIndices[i];
            }

            std::cout << std::endl;
        }
    }
}


static cl::opt<string> Basicblockfilepath("Basicblockfilepath", cl::desc("the path of Basicblockfile"), cl::value_desc("string"));

class CustomAsmInsertionPass : public FunctionPass
{
public:
  static char ID;
  MutationInfo functionsmap;

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
    // 28 对应  flatten-1,   --- 函数级别的--函数中全部基本块都做变换
    // 29 对应  flatten-2,
    // 30 对应  flatten-3,
    // 31 对应  bcf_rate  0-100,
};

  CustomAsmInsertionPass() : FunctionPass(ID) {
    string basicblockpath = Basicblockfilepath.getValue();
    functionsmap = parseMutationFile(basicblockpath);
    // printModuleMutationInfo(functionsmap);
  }

  bool runOnFunction(Function &F) override {

    if (functionsmap.find(F.getName().str()) != functionsmap.end()) {
      //输出函数名
      // std::cout <<  F.getName().str() << std::endl;

      auto& funcInfo = functionsmap[F.getName().str()];
      // 根据 funcInfo.flattenLevel 和 funcInfo.bcfRate 应用其他变异策略

      // 启动 BogusControlFlow 或 FlatPlus
      if (funcInfo.flattenLevel != 0){
        createFlatPlus(true, false, funcInfo.flattenLevel)->runOnFunction(F);
        return true;
      }

      if (funcInfo.bcfRate != 0){
        createBogusControlFlow(true, funcInfo.bcfRate)->runOnFunction(F);
        return true;
      }
      // createBogusControlFlow(flag=true, bcf_rate=funcInfo.bcfRate)->runOnFunction(F);
      // createFlatPlus(flag=true, dont_fla_invoke=false, fla_cnt=funcInfo.flattenLevel)->runOnFunction(F)

      // createFlatPlus(true, false, funcInfo.flattenLevel) -> runOnFunction(F);
      // return true;

      int index = 0; // 基本块的索引
      for (auto& BB : F) {
        if (funcInfo.blockInfos.find(index) != funcInfo.blockInfos.end()) {
          //输出基本块索引
          // std::cout <<  index << std::endl;
          
          auto& blockInfo = funcInfo.blockInfos[index];
          // 使用 blockInfo.asmInstructionsIndices 插入汇编指令
          for (int asmIndex : blockInfo.asmInstructionsIndices) {
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
        index += 1;
      }
    }

    return true;
  }
};


char CustomAsmInsertionPass::ID = 0;
static RegisterPass<CustomAsmInsertionPass> X("fuzz-asm-insertion", "Insert custom assembly instruction at the beginning of each basic block", false, false);
