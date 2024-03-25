#include "AsmInstruction.hpp"
#include "IRFile.hpp"
#include <vector>
using namespace std;

class Attack
{

public:
    /*
        NOP指令集
        针对x86 32位写的，64位虽说也能兼容这些指令。
        1. 但是pushfd这种指令不知道64位下面能不能兼容？
        2. 单个“NOP”指令还可以有很大的变化。譬如增加+1 -1的次数，这种变化是否有意义？
    */
    vector<string> asmInsructionArray = {
        "nop\n", 
        "sub eax,0\n",
        "add eax,0\n",
        "lea eax,[eax+0]\n",
        "mov eax,eax\n",
        "xchg eax,eax\n",
        "pushfd\n push eax\n xor eax, eax\n comvo eax,ecx\n pop eax\n popfd\n", //OF溢出标志
        "pushfd\n push eax\n xor eax, eax\n comvp eax,eax\n pop eax\n popfd\n", //PF奇偶校验标志
        "pushfd\n cmp eax, eax\n comva eax, eax\n popfd\n",//CF进位标志、ZF零标志 条件移动指令
        "pushfd\n cmp eax, eax\n comvg eax, ecx\n popfd\n",//ZF零标志、SF符号标志、溢出标志OF  
        "pushfd\n push eax\n mov eax, -1\n cmp eax, 0\n cmovs eax, eax\n pop eax\n pop fd\n",//SF符号标志
        "pushfd\n cmp eax, eax\n cmovl eax, ecx\n popfd\n",//小于的条件下移动数据
        "pushfd\n cmp eax, eax\n cmovns eax, eax\n popfd\n",//在符号标志未设置时移动数据
        "pushfd\n push eax\n xor eax,eax\n cmovnp eax, ecx\n pop eax\n popfd\n",//在奇偶校验标志未设置时移动数据
        "pushfd\n cmp eax, eax\n cmovno eax,ecx\n popfd\n",//溢出标志被设置时移动数据
        "add eax, 1\n sub eax,1\n",
        "sub eax, -2\n add eax, 2\n",
        "push eax\n neg eax\n neg eax\n pop eax\n",//求补码操作
        "NOT eax\n NOT eax\n",//取反操作
        "push eax\n pop eax\n",
        "pushfd\n popfd\n",//保存标志寄存器 f-16位 fd-32位 fq-64位
        "xchg eax, ecx\n xchg ecx,eax\n",
        "push eax\n not eax\n pop eax\n",
        "xor eax, ebx\n xor ebx, eax\n xor ebx, eax\n xor eax, ebx\n",
        "pop ebx\n mov ebx, eax\n add eax,1\n mov eax, ebx\n pop ebx\n",
        "push eax\n inc eax\n dec eax\n dec eax\n pop eax\n",
        "push ebx\n mov ebx, eax\n cmp eax, eax\n setg al\n movzx eax, al\n mov eax, ebx\n pop ebx\n",//setg指令根据比较结果设置条件标志
    };

    vector<AsmInstruction> asmInsructionList;
    
    IRFile irFile("Test!");//这里是文件的名字

public:
    Attack();
    ~Attack();
};

Attack::Attack(){
    // 初始化asmInstructionList
    for (size_t i = 0; i < asmInsructionArray.size(); ++i) {
        AsmInstruction asmInstruction(i,asmInsructionArray[i]);
        asmInsructionList.push_back(asmInstruction);
    }
    // 根据IRFile更新asmInsructionList中每一个asmInstruction的插入的次数
    for (const auto& function : irFile.functionList){
        for(const auto& num : function.asmInsertionList){
            asmInsructionList[num].numInsertion ++;
        }
    }

}

Attack::~Attack(){
    
}