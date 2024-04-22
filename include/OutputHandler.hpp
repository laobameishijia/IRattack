#include "llvm/IR/Function.h"
#include <string>
#include <fstream>

class OutputHandler {
public:
    std::ofstream outFile;
    std::string outputFileName;

    OutputHandler(const std::string& prefix);
    ~OutputHandler();

    void writeHeader(const llvm::Function &F, int flattenLevel, int bcfRate);
    void writeBasicBlockInfo(const llvm::Function &F, unsigned int BBCounter, unsigned int InstCounter);
};