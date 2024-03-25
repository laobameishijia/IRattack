class AsmInstruction:
    def __init__(self, number, string_instruction):
        self.string_instruction = string_instruction  # 汇编指令字符串
        self.number = number  # 汇编指令的序号
        self.num_instruction = 0  # 指令数量，初始设为0
        self.num_insertion = 0  # 用其插入基本块的次数，初始设为0
        self.insert_positions = []  # 存储插入位置的列表 _Z9ncfution1i#0 基本块的名字

    def add_insert_position(self, insert_position):
        # 将新的插入位置添加到列表中，并增加插入次数
        self.insert_positions.append(insert_position)
        self.num_insertion += 1

    def stats_num_instruction(self):
        # 计算string_instruction中换行符的数量
        self.num_instruction = self.string_instruction.count('\n')


if __name__ == "__main__":
    # 使用示例
    asm = AsmInstruction(1, "mov eax, ebx \n mov ebx, eax\n")
    asm.add_insert_position("Function_name_1")
    asm.stats_num_instruction()

    print(asm.num_instruction)  # 输出指令数量
