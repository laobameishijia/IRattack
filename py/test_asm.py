from model.src.dataset_construct.asm import *



test_dir = '/home/lebron/disassemble/attack/asm'
binary_id = 'server'
# binary_id = 'LGbDxkN6wV9TedtYchBA'

parser = AsmParser(directory=test_dir, binary_id=binary_id)
# parser.parse_instructions()
parser.parse()
# parser.print_blocks()

    

