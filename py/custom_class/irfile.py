from custom_class.irblock import IRBlock


import sys

def clean_strlist2intlist(asm_nums):
    cleaned_nums = []
    for num_str in asm_nums:
        # 去除字符串两端的空白字符（包括换行符、空格等）
        num_str = num_str.strip()

        # 检查处理后的字符串是否为空或不是数字
        if num_str and num_str.isdigit():
            # 转换为整数并添加到结果列表
            cleaned_nums.append(int(num_str))

    return cleaned_nums
        
class IRFile:
    def __init__(self, name):
        self.name = name
        self.block_list = []
        self.read_and_parse_file(filename=name)

    def read_and_parse_file(self, filename):
        """_summary_:
        读取解析basicblock.txt文件
        
        Args:
            filename (string):文件名

        Returns:
            -1 : 无法成功打开文件
            0  : 正常解析文件
        """
        try:
            with open(filename, 'r') as file:
                for line in file: # line:'_Z9function1i#0&5: +1+2+3+4\n'
                    if '#' in line:
                        parts = line.split(':')# parts[0]:_Z9ncfution1i#0&5 | parts[1]: +1+2+3+4\n
                        blockName = parts[0].split('&')[0] # _Z9ncfution1i#0
                        inst_nums = int(parts[0].split('&')[1]) # 5
                        block = IRBlock(blockName, inst_nums)
                        
                        blocks_and_asminstructions = parts[1]
                        if len(blocks_and_asminstructions) > 1:
                            asm_nums = blocks_and_asminstructions.split('+') # [' ','1','2','3','4\n']
                            asm_nums = clean_strlist2intlist(asm_nums) # ['1','2','3','4']
                            for num in asm_nums:
                                block.add_asm_instruction(int(num))

                        self.block_list.append(block)
            file.close()
            return 0
        except FileNotFoundError:
            sys.stderr.write(f"无法打开BasicBlock.txt文件: {filename}\n")
            return -1

if __name__ == "__main__":
    # 使用示例
    ir_file = IRFile("/home/lebron/IRattack/test/BasicBlock.txt")
    # ir_file.read_and_parse_file("/home/lebron/IRattack/test/BasicBlock.txt")
    print("Test")