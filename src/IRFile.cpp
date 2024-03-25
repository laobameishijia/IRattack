#include "IRFile.hpp"
#include "IRFunction.hpp"
#include <fstream>
#include <iostream>

using namespace std;

IRFile::IRFile(string name):name(name){
    readAndParseFile("BasicBlock.txt")
}

IRFile::~IRFile(){

}


void IRFile::addFunction(Function function){
    functionList.push_back(function)
}


/**
 * 读取并解析BasicBlock.txt文件
 * 
 * 格式：函数名#该函数中基本块序号:1+2+3+4
 * 其中1,2,3,4为插入的asmStringArray元素的序数。
*/
int IRFile::readAndParseFile(const string& fileName){
    ifstream file(fileName);

    if (!file) {
        cerr << "无法打开BasicBlock.txt文件: " << fileName << endl;
        return -1;
    }
    
    while (getline(file, line)) {
        stringstream ss(line);
        string functionName;
        string temp;
        
        // 读取函数名
        if (getline(ss, functionName, '#')) {
            IRFunction function;
            function.name = functionName;
            // 获取基本块序号、数量
            int num_temp;
            getline(ss, temp, ':');
            stringstream numStream(temp);
            numStream >> num_temp;
            function.addBasicBlock(num_temp);
            function.statsNumBasicBlock();
            
            // 获取添加asm指令的序号
            while (getline(ss, temp, '+')) {
                int num;
                stringstream numStream(temp);
                if (numStream >> num) {  // 只处理数字，忽略其他字符
                    function.addAsmInstruction(num);
                }
            }
            addFunction(function)
        }
    }
    file.close();
}
