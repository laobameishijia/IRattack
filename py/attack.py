import torch
from custom_class.irfile import IRFile
from custom_class.asminstruction import AsmInstruction
from actor_critic import CFGEnvironment,ActorCritic

class Attack:
    
    def __init__(self,irfilename):
        
        self.asm_instruction_array = [
            "nop\n", 
            "sub eax,0\n",
            "add eax,0\n",
            "lea eax,[eax+0]\n",
            "mov eax,eax\n",
            "xchg eax,eax\n",
            "pushfd\n push eax\n xor eax, eax\n comvo eax,ecx\n pop eax\n popfd\n", #OF溢出标志
            "pushfd\n push eax\n xor eax, eax\n comvp eax,eax\n pop eax\n popfd\n", #PF奇偶校验标志
            "pushfd\n cmp eax, eax\n comva eax, eax\n popfd\n",#CF进位标志、ZF零标志 条件移动指令
            "pushfd\n cmp eax, eax\n comvg eax, ecx\n popfd\n",#ZF零标志、SF符号标志、溢出标志OF  
            "pushfd\n push eax\n mov eax, -1\n cmp eax, 0\n cmovs eax, eax\n pop eax\n pop fd\n",#SF符号标志
            "pushfd\n cmp eax, eax\n cmovl eax, ecx\n popfd\n",#小于的条件下移动数据
            "pushfd\n cmp eax, eax\n cmovns eax, eax\n popfd\n",#在符号标志未设置时移动数据
            "pushfd\n push eax\n xor eax,eax\n cmovnp eax, ecx\n pop eax\n popfd\n",#在奇偶校验标志未设置时移动数据
            "pushfd\n cmp eax, eax\n cmovno eax,ecx\n popfd\n",#溢出标志被设置时移动数据
            "add eax, 1\n sub eax,1\n",
            "sub eax, -2\n add eax, 2\n",
            "push eax\n neg eax\n neg eax\n pop eax\n",#求补码操作
            "NOT eax\n NOT eax\n",#取反操作
            "push eax\n pop eax\n",
            "pushfd\n popfd\n",#保存标志寄存器 f-16位 fd-32位 fq-64位
            "xchg eax, ecx\n xchg ecx,eax\n",
            "push eax\n not eax\n pop eax\n",
            "xor eax, ebx\n xor ebx, eax\n xor ebx, eax\n xor eax, ebx\n",
            "pop ebx\n mov ebx, eax\n add eax,1\n mov eax, ebx\n pop ebx\n",
            "push eax\n inc eax\n dec eax\n dec eax\n pop eax\n",
            "push ebx\n mov ebx, eax\n cmp eax, eax\n setg al\n movzx eax, al\n mov eax, ebx\n pop ebx\n",#setg指令根据比较结果设置条件标志
        ]

        self.asm_instruction_list = [AsmInstruction(i, ins) for i, ins in enumerate(self.asm_instruction_array)]
        
        self.ir_file = IRFile(irfilename)
        self.initialize_asm_instruction_list()

    def initialize_asm_instruction_list(self):
        """_summary_:
            根据IRFile文件对应的basicblock.txt初始化汇编指令类, 
            计算得到每个汇编指令插入的位置(函数名),以及插入的次数。

        """
        # 初始化asm_instruction_list
        for block in self.ir_file.block_list:
            for num in block.asm_instructions:
                asm_instruction = self.asm_instruction_list[num] # AsmInstruction类
                asm_instruction.add_insert_position(insert_position=block.name)


if __name__ == "__main__":
    
    
    attack = Attack("/home/lebron/disassemble/attack/sourcecode/Linux.Apachebd/attack/BasicBlock.txt")

    basic_blocks_info = dict()
    for block in attack.ir_file.block_list:
        inst_nums = block.inst_nums
        nops_insert_count = {i: 0 for i in range(27)}
        for nop_id in block.asm_instructions:
            nops_insert_count[nop_id] += 1
        basic_blocks_info[block.name]  = {
            "inst_nums":inst_nums,
            "nops_insert_count":nops_insert_count
        }
    
    all_nops_insert_count = {i: 0 for i in range(27)}
    
    for nop_id, nop_asm in enumerate(attack.asm_instruction_list):
        all_nops_insert_count[nop_id] += nop_asm.num_insertion
        
    # train(basic_blocks_info=basic_blocks_info, nop_lists=all_nops_insert_count)
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    gamma = 0.98
    device = torch.device("cpu")

    Test = ActorCritic(basic_blocks_info,all_nops_insert_count, actor_lr,critic_lr,gamma,device)
    Test.train(num_episodes)
    print("Test")
    
    