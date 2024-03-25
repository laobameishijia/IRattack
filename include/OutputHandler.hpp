#include "llvm/IR/Function.h"
#include <string>
#include <fstream>

class OutputHandler {
public:
    OutputHandler();  // 空的构造函数
    explicit OutputHandler(const std::string& prefix);
    ~OutputHandler();

    void writeBasicBlockInfo(const llvm::Function &F, unsigned int BBCounter,unsigned int InstCounter);

private:
    std::string outputFileName;
    std::ofstream outFile;  // 将ofstream作为成员变量
};
