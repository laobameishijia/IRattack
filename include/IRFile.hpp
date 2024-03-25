#include <vector>
#include <string>
#include "IRFunction.hpp"
/**
 * 中间语言文件 -- 实际上是中间语言输出的basicBlock.txt
 * 
 * 中间语言文件 1:N 函数
*/
class IRFile
{
public:
    vector<IRFunction> functionList;// 函数列表
    string name;
public:
    IRFile(string name);
    ~IRFile();

    void addFunction(IRFunction function);
    int readAndParseFile(const string& fileName);

};