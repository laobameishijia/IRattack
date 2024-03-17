#include <iostream>
#include <vector>
#include <string>


using namespace std;

/**
 * 插入位置的结构体
 * 包含函数名、基本块id、插入asmInstruction的id
*/
struct InsertPostion
{
    string functionName;
    int block_id;
    vector<int> asmInsruction_id;
};

class AsmInstruction{

public:

// NOP指令集
/*
针对x86 32位写的，64位虽说也能兼容这些指令。
  1. 但是pushfd这种指令不知道64位下面能不能兼容？
  2. 单个“NOP”指令还可以有很大的变化。譬如增加+1 -1的次数，这种变化是否有意义？
*/
vector<string> asmStringArray = {
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

// 读取并解析BasicBlock.txt文件
/**
 * 格式：函数名#该函数中基本块序号:1+2+3+4
 * 其中1,2,3,4为插入的asmStringArray元素的序数。
*/
vector<InsertPostion> readAndParseFile(const string& fileName) {
    vector<InsertPostion> result;
    ifstream file(fileName);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string functionName;
        string temp;
        
        // 读取函数名
        if (getline(ss, functionName, '#')) {
            InsertPostion data;
            data.functionName = functionName;
            
            int num_temp;
            getline(ss, temp, ':');
            stringstream numStream(temp);
            numStream >> num_temp;
            data.block_id = num_temp;
            
            // 读取后续的数字
            while (getline(ss, temp, '+')) {
                int num;
                stringstream numStream(temp);
                if (numStream >> num) {  // 只处理数字，忽略其他字符
                    data.asmInsruction_id.push_back(num);
                }
            }

            result.push_back(data);
        }
    }

    file.close();
    return result;
}

}

int main() {
    std::string fileName = "BasicBlock.txt";
    auto parsedData = readAndParseFile(fileName);

    // 输出解析结果
    for (const auto& data : parsedData) {
        std::cout << "Function Name: " << data.functionName << std::endl;
        std::cout << "Numbers: ";
        for (const auto& num : data.numbers) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}