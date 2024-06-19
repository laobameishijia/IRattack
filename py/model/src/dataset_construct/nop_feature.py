import json

# 27条语义NOP指令集
semantic_nops = [
    "nop\n",
    "subq $$0x0, %rax\n",
    "addq $$0x0, %rax\n",
    "leaq (%rax), %rax\n",
    "movq %rax, %rax\n",
    "xchgq %rax, %rax\n",
    "pushfq\n pushq %rax\n xorl %eax, %eax\n cmovol %ecx, %eax\n popq %rax\n popfq\n",
    "pushfq\n pushq %rax\n xorl %eax, %eax\n cmovpl %eax, %eax\n popq %rax\n popfq\n",
    "pushfq\n cmpq %rax, %rax\n cmovb %eax, %eax\n popfq\n",
    "pushfq\n cmpq %rax, %rax\n cmovg %ecx, %eax\n popfq\n",
    "pushfq\n cmpq %rax, %rax\n cmovs %ecx, %eax\n popfq\n",
    "pushfq\n cmpq %rax, %rax\n cmovl %ecx, %eax\n popfq\n",
    "pushfq\n cmpq %rax, %rax\n cmovns %eax, %eax\n popfq\n",
    "pushfq\n pushq %rax\n xorl %eax, %eax\n cmovnp %ecx, %eax\n popq %rax\n popfq\n",
    "pushfq\n cmpq %rax, %rax\n cmovno %ecx, %eax\n popfq\n",
    "addq $$0x1, %rax\n subq $$0x1, %rax\n",
    "subq $$-2, %rax\n addq $$0x2, %rax\n",
    "pushq %rax\n negq %rax\n negq %rax\n popq %rax\n",
    "notq %rax\n notq %rax\n",
    "pushq %rax\n popq %rax\n",
    "pushfq\n popfq\n",
    "xchgq %rax, %rcx\n xchgq %rcx, %rax\n",
    "pushq %rax\n notq %rax\n popq %rax\n",
    "xorq %rbx, %rax\n xorq %rax, %rbx\n xorq %rax, %rbx\n xorq %rbx, %rax\n",
    "pushq %rbx\n movq %rax, %rbx\n addq $$0x1, %rax\n movq %rbx, %rax\n popq %rbx\n",
    "pushq %rax\n incq %rax\n decq %rax\n decq %rax\n popq %rax\n",
    "pushq %rbx\n movq %rax, %rbx\n cmpq %rax, %rax\n setg %al\n movzbq %al, %rax\n movq %rbx, %rax\n popq %rbx\n"
]

# 生成控制流图
cfg = {}
start_address = 1000
for i, nop in enumerate(semantic_nops):
    end_address = start_address + len(nop.split('\n')) - 1  # 每条指令占用一个地址空间
    instructions = []
    current_address = start_address
    for line in nop.strip().split('\n'):
        parts = line.split()
        opcode = parts[0]
        operands = parts[1:] if len(parts) > 1 else []
        instructions.append({
            "address": current_address,
            "opcode": opcode,
            "operands": operands,
            "next_addr": current_address + 1
        })
        current_address += 1
    
    out_edge_list = []
    if start_address + 1000 < 1000 + 27000:
        out_edge_list.append(start_address + 1000)

    basic_block = {
        "start_addr": start_address,
        "end_addr": end_address,
        "insn_list": instructions,
        "in_edge_list": [start_address - 1000] if start_address > 1000 else [],
        "out_edge_list": out_edge_list
    }
    cfg[str(start_address)] = basic_block
    start_address += 1000

# 打印控制流图
print(json.dumps(cfg, indent=4))
