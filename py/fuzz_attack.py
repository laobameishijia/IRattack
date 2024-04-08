import os
import random
import shutil
import copy
import torch
from termcolor import colored

from model.src.dataset_construct import run_disassemble
from model.src.dataset_construct import run_extract_cfg
from model.src.gnn import run_measure

class BlockInfo:
    def __init__(self, count=0, asm_indices=None):
        self.count = count
        self.asm_indices = asm_indices if asm_indices is not None else []

class FunctionInfo:
    def __init__(self):
        self.blocks = {}

def parse_asm_indices(line):
    """解析字符串中所有+号后面的数字，并添加到列表中"""
    return [int(num) for num in line.split('+')[1:]]

def parse_file(file_path):
    """解析文件并填充包含多个函数信息的字典"""
    functions = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            functionName, rest = line.split('#')
            blockNum_str, asmPart = rest.split('&')
            blockNum = int(blockNum_str)
            count_str, _, asmIndices_str = asmPart.partition(': ')
            count = int(count_str.strip(': '))  # 修改此处，确保从字符串提取整数前去除了冒号和空格
            
            asmIndices = []
            if asmIndices_str.strip():  # 确保不是空字符串
                asmIndices = parse_asm_indices(asmIndices_str)
            
            if functionName not in functions:
                functions[functionName] = FunctionInfo()
            functions[functionName].blocks[blockNum] = BlockInfo(count, asmIndices)
    return functions

def output_file(functions, output_path):
    """按照原始文件格式输出内容到文件"""
    with open(output_path, 'w') as file:
        for functionName, functionInfo in functions.items():
            for blockNum, blockInfo in functionInfo.blocks.items():
                line = f"{functionName}#{blockNum}&{blockInfo.count}: "
                if blockInfo.asm_indices:
                    line += '+' + '+'.join(map(str, blockInfo.asm_indices))
                file.write(line + '\n')

def run_bash(script_path, args:list):
    import subprocess
    print(f"Now run {script_path}")
    # 运行脚本
    result = subprocess.run([script_path] + args, text=True)
    # 等待子进程结束  打印脚本的输出
    # result.wait()
    print(f"exit {script_path}")
    
def build_fuzz_directories(fuzz_dir):
    # 要检查和创建的文件夹列表
    folders = ["in", "out", "rawdata", "asm", "cfg", "cfg_magic_test","temp"]

    for folder in folders:
        # 构造完整的文件夹路径
        folder_path = os.path.join(fuzz_dir, folder)
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 文件夹不存在，创建它
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            # 文件夹已存在
            print(f"Folder already exists: {folder_path}")

def list_txt_files(directory):
    """
    列出指定目录下所有的.txt文件的完整路径。

    :param directory: 目录的路径。
    :return: 一个包含所有.txt文件完整路径的列表。
    """
    txt_files = []  # 用于存储找到的.txt文件的完整路径

    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录不存在：{directory}")
        return txt_files

    # 获取目录下所有内容
    items = os.listdir(directory)

    # 过滤出.txt文件，并添加完整路径
    for item in items:
        full_path = os.path.join(directory, item)
        if item.endswith('.txt') and os.path.isfile(full_path):
            txt_files.append(full_path)

    return txt_files

def copy_file_to_folder(source_file, target_folder):
    """
    将指定的文件复制到指定的文件夹。

    :param source_file: 要复制的文件的路径。
    :param target_folder: 目标文件夹的路径。
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"目标文件夹不存在，已创建：{target_folder}")
    
    # 构建目标文件的完整路径
    target_file_path = os.path.join(target_folder, os.path.basename(source_file))
    
    # 复制文件
    try:
        shutil.copy(source_file, target_file_path)
        print(f"文件已成功复制到：{target_file_path}")
    except IOError as e:
        print(f"无法复制文件。{e}")

def disassemble(fuzz_dir):
    dataset_dir= fuzz_dir
    dir_path = f"{dataset_dir}/rawdata"
    output_dir = f"{dataset_dir}/asm"
    log_path = f"{dataset_dir}/disassemble.log"
    run_disassemble.run(dir_path=dir_path,output_dir=output_dir,log_path=log_path)

def extract_cfg(fuzz_dir):
    dataset_dir= fuzz_dir
    data_dir = f"{dataset_dir}/asm"
    store_dir = f"{dataset_dir}/cfg"
    file_format = "json"
    log_file = f"{dataset_dir}/extract_cfg.log"
    run_extract_cfg.run(data_dir=data_dir,store_dir=store_dir,file_format=file_format,log_file=log_file)

def measure(fuzz_dir):
    data_dir = fuzz_dir
    result = run_measure.measure(data_dir=data_dir)
    return result

def add_random_asmIndex(functions, changes_log):
    # 随机挑选一个函数名
    max_log_prob = 0.8
    log_len = len(changes_log)
    # 计算从changes_log中挑选的概率，但不超过max_log_prob = 0.8
    prob = min(log_len / (log_len + 100), max_log_prob)  # 示例逻辑，可根据需要调整

    if random.random() < prob and changes_log:
        # 以一定概率从changes_log中选择
        change = random.choice(changes_log)
        functionName = change["functionName"]
        blockNum = change["blockNum"]
        asmIndex = change["asmIndex"]
        print(f"Selected - Function: {functionName}, Block: {blockNum}, asmIndex: {asmIndex}")
        functions[functionName].blocks[blockNum].asm_indices.append(asmIndex)
        return {}
    else:
        # 随机挑选一个函数
        functionName = random.choice(list(functions.keys()))
        # 随机挑选该函数下的一个基本块
        blockNum = random.choice(list(functions[functionName].blocks.keys()))
        # 生成一个范围在0-26之间的随机asmIndex
        asmIndex = random.randint(0, 26)
        print(f"Selected - Function: {functionName}, Block: {blockNum}, asmIndex: {asmIndex}")
        # 将随机生成的asmIndex添加到选定的基本块的asmIndices列表中
        functions[functionName].blocks[blockNum].asm_indices.append(asmIndex)
        return {"functionName": functionName, "blockNum": blockNum, "asmIndex": asmIndex}

    # # 返回更新后的functions
    # return functionName

def main():
    file_path = "/home/lebron/disassemble/attack/sourcecode/Linux.Apachebd/attack/BasicBlock.txt"
    output_path = "/home/lebron/IRattack/xuange"
    functions = parse_file(file_path)
    
    # 示例输出
    for functionName, functionInfo in functions.items():
        print(f"Function: {functionName}")
        for blockNum, blockInfo in functionInfo.blocks.items():
            print(f"  Block #{blockNum}, Count: {blockInfo.count}, Asm Indices: {' '.join(map(str, blockInfo.asm_indices))}")

    # 输出到文件
    # output_file(functions, output_path)
    
    source_dir="/home/lebron/disassemble/attack/sourcecode/Linux.Apachebd/attack"
    fuzz_dir="/home/lebron/IRFuzz/Linux.Apachebd"
    bash_sh = "/home/lebron/disassemble/attack/sourcecode/Linux.Apachebd/attack/fuzz_insertasminstruction.sh"
    temp_bb_file_path = f"{fuzz_dir}/temp/.basicblock"
   
    build_fuzz_directories(fuzz_dir)
    copy_file_to_folder(source_file=f"{source_dir}/BasicBlock.txt",target_folder=f"{fuzz_dir}/in")
    
    seed_list = list_txt_files(directory=f"{fuzz_dir}/in")
    
    previous_probability_0 = 0
    seed_count = 0
    
    for basicblockfile in seed_list:
        
        num_episodes = 1000
        functions = parse_file(basicblockfile) # 解析文件
        copy_functions = copy.deepcopy(functions) # 保存原有副本
        changes_log = [] # 保留之前让文件良性概率增加的操作
        
        while num_episodes > 0:
            # 第一次进来,先不变异.为了获取初始恶意类别的分类概率
            if num_episodes == 1000:
                num_episodes -= 1
                output_file(functions, temp_bb_file_path)
                # 插入+链接
                run_bash(script_path= bash_sh,
                        args=[source_dir, fuzz_dir, temp_bb_file_path])
                # 返汇编
                disassemble(fuzz_dir=fuzz_dir)
                # 提取cfg
                extract_cfg(fuzz_dir=fuzz_dir)
                # 模型预测
                next_state, result, prediction = measure(fuzz_dir) # prediction 0是良性 1是恶意  目前要把恶意转为良性。 result是模型输出的logsoftmax概率
                result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
                formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)
                probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
                probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
                previous_probability_0 = probability_0 # 保留初始的概率
                previous_probability_1 = probability_1 # 保留初始的概率
                print(colored(f"probability_0: {probability_0}  probability_1:{probability_1}", 'green'))

                continue
            else:  
                num_episodes -= 1
                change = add_random_asmIndex(functions,changes_log) #增加随机变异,用change保留下来
                output_file(functions, temp_bb_file_path)
                # 插入+链接
                run_bash(script_path= bash_sh,
                        args=[source_dir, fuzz_dir, temp_bb_file_path])
                # 返汇编
                disassemble(fuzz_dir=fuzz_dir)
                # 提取cfg
                extract_cfg(fuzz_dir=fuzz_dir)
                # 模型预测
                next_state, result, prediction = measure(fuzz_dir) # prediction 0是良性 1是恶意  目前要把恶意转为良性。 result是模型输出的logsoftmax概率
                result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
                formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)

                probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
                probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
                print(colored(f"probability_0: {probability_0}  probability_1:{probability_1}", 'green'))
                
                # 如果良性的概率增加,将当前变异策略输出到out目录当中
                if  probability_0 > previous_probability_0: 
                    # 刚开始只要比最初始的概率大，就保存。
                    # 当changes_logs到达50个的时候，那么提升不大的操作就不保存。
                    if len(changes_log) > 50:
                        previous_probability_0 = probability_0
                    output_file(functions, f"{fuzz_dir}/out/{seed_count}")
                    seed_count += 1
                    print(colored(f"save functions to {fuzz_dir}/out/{seed_count}", 'green'))
                    
                    if change is not {}: # 如果返回的操作不为空，将其保存在changes_log中
                        changes_log.append(change)
                        
                # 如果没有这种变化,就舍弃当前给出的随机变化,重新将functions设置为原始
                else:
                    functions=copy_functions 

            

if __name__ == "__main__":
    main()

