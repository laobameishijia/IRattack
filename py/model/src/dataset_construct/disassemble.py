import idautils
import idc
import ida_funcs
import ida_segment
import ida_bytes
import ida_auto

import idaapi



def is_arm_architecture():
    # 获取当前二进制文件的处理器类型
    info = idaapi.get_inf_structure()
    # IDA Pro中定义的处理器名称可以在idaapi.py或相应的处理器模块文档中找到
    if info.procname.lower().startswith("arm"):
        print("This binary is based on ARM architecture.")
        return True
    else:
        print("This binary is not based on ARM architecture.")
        return False
    
def save_disassembly(output_file_path, file_name, log_file):
    # 设置保存文件的路径
    ida_auto.auto_wait()  # 等待IDA完成自动分析
    if not is_arm_architecture(): # 暂时先不考虑arm架构
        log_file.write(f"{file_name} is x86\n")
        log_file.flush()  # 强制刷新缓冲区，写入文件
        with open(output_file_path, "w") as file:
            for seg_ea in idautils.Segments():
                seg = ida_segment.getseg(seg_ea)  # 获取segment_t对象
                if seg and seg.perm & ida_segment.SEGPERM_READ and seg.perm & ida_segment.SEGPERM_EXEC:
                    seg_name = ida_segment.get_segm_name(seg)  # 获取段名
                    start = seg.start_ea
                    end = seg.end_ea
                    ea = start
                    while ea < end:
                        disasm_line = idc.generate_disasm_line(ea, 0)
                        bytes = ida_bytes.get_bytes(ea, ida_bytes.get_item_size(ea))
                        bytes_str = ' '.join(['{:02X}'.format(b) for b in bytes])
                        if disasm_line:
                            file.write(f"{seg_name}:{hex(ea)[2:].upper()} {bytes_str.ljust(20)} {disasm_line}\n")
                        ea = idc.next_head(ea, end)
    else:
        log_file.write(f"{file_name} is arm\n")
        log_file.flush()  # 强制刷新缓冲区，写入文件
    idc.qexit(0)


def save_disassembly_formatted(output_file_path):
    ida_auto.auto_wait() # 等待ida分析完成
    with open(output_file_path, 'w', encoding='utf-8') as f:  # 指定编码为UTF-8
        for segea in idautils.Segments():
            seg = ida_segment.getseg(segea)
            if seg:
                segname = ida_segment.get_segm_name(seg)
                if segname == '.text':
                    for funcea in idautils.Functions(segea, idc.get_segm_end(segea)):
                        functionName = ida_funcs.get_func_name(funcea)
                        f.write(".text:{}                               ; =============== S U B R O U T I N E =======================================\n".format(hex(funcea)))
                        f.write(".text:{}\n".format(hex(funcea)))
                        f.write(".text:{}\n".format(hex(funcea)))
                        f.write(".text:{}                               {} proc near\n".format(hex(funcea), functionName))
                        for (startea, endea) in idautils.Chunks(funcea):
                            for head in idautils.Heads(startea, endea):
                                disasm = idc.generate_disasm_line(head, 1)
                                bytes = ida_bytes.get_bytes(head, idc.get_item_size(head))
                                bytes_str = ' '.join(['{:02X}'.format(b) for b in bytes])
                                f.write(".text:{:08X} {:<25} {}\n".format(head, bytes_str, disasm))
                        f.write(".text:{}\n".format(hex(funcea)))
                        f.write("\n")
    idc.qexit(0)

def save_disassembly_from_start(output_file_path):
    ida_auto.auto_wait()  # 正确的等待分析完成的调用

    # 找到程序的开始地址
    start_address = idaapi.get_imagebase()
    
    # 找到程序的结束地址
    end_address = max([ida_segment.getseg(segea).end_ea for segea in idautils.Segments()])

    with open(output_file_path, 'w', encoding='utf-8') as f:
        current_address = start_address
        while current_address < end_address:
            disasm = idc.generate_disasm_line(current_address, 1)
            if disasm:
                bytes = ida_bytes.get_bytes(current_address, idc.get_item_size(current_address))
                if bytes:  # 确保地址处有指令或数据
                    bytes_str = ' '.join(['{:02X}'.format(b) for b in bytes])
                    f.write(".text:{:08X} {:<25} {}\n".format(current_address, bytes_str, disasm))
            # 移动到下一个头，无论是数据还是指令
            current_address = idc.next_head(current_address, end_address)

    # idc.qexit(0)

def save_disassembly_for_all_segments_with_segment_name(output_file_path):
    ida_auto.auto_wait()  # 等待IDA完成自动分析

    with open(output_file_path, 'w', encoding='utf-8') as f:
        # 遍历所有段
        for segea in idautils.Segments():
            seg = ida_segment.getseg(segea)
            segname = ida_segment.get_segm_name(seg)
            segstart = seg.start_ea
            segend = seg.end_ea
            
            if segname != ".text": break # 只分析.text段
            f.write("; =============== Segment: {}, Start: {}, End: {} ===============\n".format(segname, hex(segstart), hex(segend)))

            # 遍历段内的所有头（heads）
            current_address = segstart
            while current_address < segend and current_address != idaapi.BADADDR:
                disasm = idc.generate_disasm_line(current_address, 1)
                bytes = ida_bytes.get_bytes(current_address, idc.get_item_size(current_address))
                if bytes:  # 确保地址处有指令或数据
                    bytes_str = ' '.join(['{:02X}'.format(b) for b in bytes])
                    # 在这里修改输出格式，包含段名和地址
                    f.write("{}:{:08X} {:<25} {}\n".format(segname, current_address, bytes_str, disasm))
                current_address = idc.next_head(current_address, segend)

    # idc.qexit(0)
    
# 示例用法，确保先手动用IDA打开文件
if __name__ == "__main__":
    # Test
    # output_file_path = r"E:\dataset\begnin\x86__64__lsb__unix-system-v__clang-3.8.0__O0__no-obf__unstripped__acpid-2.0.31__acpid.asm"
    # log_file = open("E:\dataset\log.txt","w")
    # file_name = "test"
    # print(log_file.write("Test is x86\n"))
    # save_disassembly(output_file_path, file_name, log_file)
    # log_file.close()
    # # # save_disassembly_formatted(output_file_path)
    # # # save_disassembly_from_start(output_file_path)
    # # # save_disassembly_for_all_segments_with_segment_name(output_file_path)
    # # exit()
    
    # Run
    import os
    output_dir = idc.ARGV[1]
    file_name = idc.ARGV[2]
    # file_name = os.path.splitext(os.path.basename(file_name))[0]
    file_name = file_name
    output_file_path = output_dir + "\\" + file_name + ".asm"
    log_file = open("/home/lebron/log","a")
    save_disassembly(output_file_path, file_name, log_file)
    log_file.close()
    exit()
