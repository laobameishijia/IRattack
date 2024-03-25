#include "IRFunction.hpp"

using namespace std;

IRFunction::IRFunction(string name):name(name){
    
}
IRFunction::statsNumBasicBlock(){
    numBasicBlock = basicBlockList.size();
}

IRFunction::addBasicBlock(int basicBlockNum){
    basicBlockList.push_back(basicBlockNum);
}

IRFunction::addAsmInstruction(int asmInstruction){
    asmInsertionList.push_back(asmInstruction);
}