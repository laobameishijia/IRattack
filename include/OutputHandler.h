#ifndef OUTPUT_HANDLER_H
#define OUTPUT_HANDLER_H

#include "llvm/IR/Function.h"
#include <string>
#include <fstream>

class OutputHandler {
public:
    explicit OutputHandler(const std::string& prefix);
    ~OutputHandler();

    void writeBasicBlockInfo(const llvm::Function &F, unsigned int BBCounter);

private:
    std::string outputFileName;
    std::ofstream outFile;  // 将ofstream作为成员变量
};

#endif // OUTPUT_HANDLER_H
