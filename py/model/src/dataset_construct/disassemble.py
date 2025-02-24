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

# 示例用法，确保先手动用IDA打开文件
if __name__ == "__main__":

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
