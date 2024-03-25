#include<string>
using namespace std;

class AsmInstruction
{
public:
    string stringInstruction;//汇编指令字符串
    int number; //汇编指令的序号
    int numInstruction;// 指令数量
    int numInsertion;// 用其插入基本块的次数
public:
    AsmInstruction(int number, string stringInstruction);
    ~AsmInstruction();

   void addInsertPosition(Function insertPosition);
   void statsNumInstruction();
};

