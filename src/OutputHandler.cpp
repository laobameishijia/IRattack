// OutputHandler.cpp

#include "OutputHandler.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

OutputHandler::OutputHandler(const std::string& prefix) {
    outputFileName = prefix.empty() ? "defaultPrefix.txt" : prefix + "BasicBlock.txt";
    outFile.open(outputFileName, std::ios_base::out);
    if (!outFile) {
        errs() << "Error: Could not open output file " << outputFileName << "\n";
    }
}

OutputHandler::~OutputHandler() {
    if (outFile.is_open()) {
        outFile.close();
    }
}

void OutputHandler::writeHeader(const Function &F, int flattenLevel, int bcfRate) {
    if (outFile.is_open()) {
        outFile << "[" << F.getName().str() << "@" << flattenLevel << "," << bcfRate << "]\n";
    } else {
        errs() << "Error: Output file is not open for writing\n";
    }
}

void OutputHandler::writeBasicBlockInfo(const Function &F, unsigned int BBCounter, unsigned int InstCounter) {
    if (outFile.is_open()) {
        outFile << F.getName().str() << "#" << BBCounter << "&" << InstCounter << ":\n";
    } else {
        errs() << "Error: Output file is not open for writing\n";
    }
}