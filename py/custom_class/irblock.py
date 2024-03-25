class IRBlock:
    def __init__(self, name, inst_nums):
        self.name = name
        self.asm_instructions = [] # 插入的汇编指令序数 [1,2,3,4] 1,2,3,4是序号
        self.inst_nums = inst_nums # 基本块中包含的指令数量
    
    def add_asm_instruction(self, num_asm):
        self.asm_instructions.append(num_asm)
    
    def stas_num_asm_instructions(self):
        return len(self.asm_instructions)