#include<vector>
#include "AsmInstruction.hpp"


class IRFunction
{
public:
    string name;
    vector<int> basicBlockList;
    int numBasicBlock;
    vector<int> asmInsertionList;
public:
    IRFunction(string name);
    ~IRFunction();
 
    addBasicBlock(int basicBlockNum);
    addAsmInstruction(int asmInstruction);
    statsNumBasicBlock();
};