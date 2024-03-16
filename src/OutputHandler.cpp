// 定义一个类来处理文件输出
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <string>
#include "llvm/IR/Function.h"


using namespace llvm;
using namespace std;


class OutputHandler {
public:
    explicit OutputHandler(const std::string& prefix) {
        outputFileName = prefix.empty() ? "defaultPrefixBasicBlock.txt" : prefix + "BasicBlock.txt";
        outFile.open(outputFileName, ios_base::out);  // 打开文件,覆盖上一次
        if (!outFile) {
            errs() << "Error: Could not open output file " << outputFileName << "\n";
        }
    }

    ~OutputHandler() {
        if (outFile.is_open()) {
            outFile.close();  // 在析构函数中关闭文件
        }
    }

    void writeBasicBlockInfo(const Function &F, unsigned int BBCounter) {
        if (outFile.is_open()) {
            outFile << "Basic Block #" << BBCounter << ": " << F.getName().str() << "\n";
        } else {
            errs() << "Error: Output file is not open for writing\n";
        }
    }

private:
    std::string outputFileName;
    std::ofstream outFile;  // 将ofstream作为成员变量
};


