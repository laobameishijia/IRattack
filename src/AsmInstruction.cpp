#include "AsmInstruction.hpp"

using namespace std;

AsmInstruction::AsmInstruction(int number, string stringInstruction):number(number),stringInstruction(stringInstruction){
    statsNumInstruction();
}
AsmInstruction::~AsmInstruction(){

}

AsmInstruction::addInsertPosition(Function insertPosition){
    insertPositionList.push_back(insertPosition);
}

AsmInstruction::statsNumInstruction(){
    int count = 0;
    for (char ch : stringInstruction) {
        if (ch == '\n') {
            count++;
        }
    }
    numInstruction = count;
}

// vector<InsertPostion> readAndParseFile(const string& fileName) {
//     vector<InsertPostion> result;
//     ifstream file(fileName);
//     string line;

//     while (getline(file, line)) {
//         stringstream ss(line);
//         string functionName;
//         string temp;
        
//         // 读取函数名
//         if (getline(ss, functionName, '#')) {
//             InsertPostion data;
//             data.functionName = functionName;
            
//             int num_temp;
//             getline(ss, temp, ':');
//             stringstream numStream(temp);
//             numStream >> num_temp;
//             data.block_id = num_temp;
            
//             // 读取后续的数字
//             while (getline(ss, temp, '+')) {
//                 int num;
//                 stringstream numStream(temp);
//                 if (numStream >> num) {  // 只处理数字，忽略其他字符
//                     data.asmInsruction_id.push_back(num);
//                 }
//             }

//             result.push_back(data);
//         }
//     }

//     file.close();
//     return result;
// }
// int main() {
//     AsmInstruction test = AsmInstruction();
   
//     std::string fileName = "/home/lebron/IRattack/test/BasicBlock.txt";
//     auto parsedData = test.readAndParseFile(fileName);

//     // 输出解析结果
//     for (const auto& data : parsedData) {
//         std::cout << "Function Name: " << data.functionName << std::endl;
//         std::cout << "BasicBlockID: " << data.block_id << std::endl;

//         std::cout << "Numbers: ";
//         for (const auto& num : data.asmInsruction_id) {
//             std::cout << num << " ";
//         }
//         std::cout << std::endl;
//     }

//     return 0;
// }