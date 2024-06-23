import argparse
import datetime
import glob
import json
import os
import random
import re
import shutil
import copy
import subprocess
import time
import torch
from termcolor import colored
from tqdm import tqdm

from model.src.dataset_construct import run_disassemble
from model.src.dataset_construct import run_extract_cfg
from model.src.gnn import run_measure
import hashlib

from model.src.gnn.model import DGCNN, GIN0, GIN0WithJK
from torch_geometric.loader import DataLoader
from model.src.gnn.dataset import CFGDataset_Semantics_Preseving, CFGDataset_MAGIC,CFGDataset_MAGIC_Attack

class BlockInfo:
    def __init__(self, count, asm_indices):
        self.count = count
        self.asm_indices = asm_indices

class FunctionInfo:
    def __init__(self, flatten_level=0, bcf_rate=0):
        self.flatten_level = flatten_level # 平坦化多少次
        self.bcf_rate = bcf_rate # 基本块作虚拟控制流的概率
        self.blocks = {}

class SeedFile:
    def __init__(self, path):
        self.path = path
        self.energy = 3
        self.asm_insert_count = 0
        self.flatten_count = 0
        self.bcf_count = 0
        self.load_and_parse()

    def update_energy(self, probability_adversarial):
        # 基于概率接近0.5的程度，以及变异操作次数计算能量
        base_energy = (probability_adversarial / 0.5 ) * 10  # 距离越近，能量越大
        # time_factor = max(0.1, 1 - (iteration / MAX_ITERATIONS))  # 随时间减少分配给距离较远的种子的能量
        # self.energy = base_energy * time_factor
        self.energy = base_energy
                   
    def load_and_parse(self):
        functions = parse_file(self.path)
        # 遍历 functions 中的所有 FunctionInfo 来计算各种操作的次数
        for func_info in functions.values():
            # 如果flatten_level不为0, 说明该函数做过一次flatten
            if func_info.flatten_level: 
                self.flatten_count += 1
            # 如果bcf_count不为0, 说明该函数做过一次bcf
            if func_info.bcf_rate:
                self.bcf_count += 1
            for block in func_info.blocks.values():
                self.asm_insert_count += len(block.asm_indices)

def parse_asm_indices(indices_str):
    """解析汇编指令索引字符串，返回索引列表"""
    return [int(idx) for idx in indices_str.split('+') if idx.strip()]

def parse_file(file_path):
    """解析文件并填充包含多个函数信息的字典，现在包括Flatten层级和BCF比例"""
    functions = {}
    current_function = None

    with open(file_path, 'r') as file:
        for line in file:
            if not line.strip():#空行跳过
                continue
            line = line.strip()
            if line.startswith('['):  # 新函数开始
                header = line.strip('[]')
                function_name, settings = header.split('@')
                flatten_level, bcf_rate = map(int, settings.split(','))
                current_function = function_name
                functions[current_function] = FunctionInfo(flatten_level, bcf_rate)
            else:
                function_name, rest = line.split('#')
                block_num_str, asm_part = rest.split('&')
                block_num = int(block_num_str)
                count_str, _, asm_indices_str = asm_part.partition(':')
                count = int(count_str.strip(':'))  # 修改此处，确保从字符串提取整数前去除了冒号和空格

                asm_indices = []
                if asm_indices_str.strip():  # 确保不是空字符串
                    asm_indices = parse_asm_indices(asm_indices_str)

                if function_name not in functions:
                    functions[function_name] = FunctionInfo()
                functions[function_name].blocks[block_num] = BlockInfo(count, asm_indices)
    file.close()
    
    return functions

def output_file(functions, output_path):
    """按照扩展文件格式输出内容到文件, 包括函数的 Flatten 级别和 BCF 比例"""
    with open(output_path, 'w') as file:
        for functionName, functionInfo in functions.items():
            # 写入函数级别的配置信息
            file.write(f"[{functionName}@{functionInfo.flatten_level},{functionInfo.bcf_rate}]\n")
            
            for blockNum, blockInfo in functionInfo.blocks.items():
                line = f"{functionName}#{blockNum}&{blockInfo.count}:" # :后面别加空格
                if blockInfo.asm_indices:
                    line += '+' + '+'.join(map(str, blockInfo.asm_indices))
                file.write(line + '\n')
            # 每个函数信息之后添加一个空行以增加可读性
            file.write('\n')
    file.close()
    
def run_bash(script_path, args, max_retries=5, retry_delay=5):
    retry_count = 0
    while retry_count < max_retries:
        try:
            result = subprocess.run([script_path] + args, timeout=30, text=True)
            if result.returncode == 0:
                return 0  # 成功运行
            else:
                print(f"Script failed with return code {result.returncode}. Retrying...")
        except subprocess.TimeoutExpired:
            print("The script timed out! Retrying...")

        retry_count += 1
        time.sleep(retry_delay)  # 等待一段时间后重试
    
    print("Max retries reached. Script failed to run successfully.")
    return -1  # 超过最大重试次数，仍然失败

def build_fuzz_directories(fuzz_dir, save_seedfile_before=False):
    # 要检查和创建的文件夹列表
    folders = ["in", "out","rawdata", "asm", "cfg","temp"]

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
    
    directory_path = os.path.join(fuzz_dir,"in")
    if os.path.exists(directory_path) and not save_seedfile_before:
        print(f"delete seedfile saved!\n")
        for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path) # 删除文件

                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    # 复制 basicblockcount.sh 和 fuzz_compile.sh到fuzz_dir目录下面
    shutil.copy("/root/IRFuzz/bash/basicblockcount.sh", fuzz_dir)
    shutil.copy("/root/IRFuzz/bash/fuzz_compile.sh", fuzz_dir)
    # 运行basicblockcount.sh 输出basicblock.txt
    run_bash(f"{fuzz_dir}/basicblockcount.sh", args=[])
    
def list_seed_files(directory):
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

def measure(fuzz_dir, model="dgcnn"):
    data_dir = fuzz_dir
    result = run_measure.measure(data_dir=data_dir, model_name=model)
    return result

def file_hash(filename):
    with open(filename, "rb") as f:
        file_contents = f.read()
    return hashlib.sha256(file_contents).hexdigest()

def is_file_duplicate(seed_order, fuzz_dir, file_hashes):

    """判断文件内容是否跟之前保存种子的重复

    Returns:
        不重复 返回 False，且把hash保存在file_hashes中
        重复   返回 True
    """
    temp_dir = f"{fuzz_dir}/temp"
    store_dir = f"{fuzz_dir}/in"
    
    file_path = f"{temp_dir}/.basicblock"
    content_hash = file_hash(file_path)
    if content_hash not in file_hashes:
        file_hashes[content_hash] = f"{store_dir}/{seed_order}.txt"
        return False
    else:
        return True

def save_hashes_to_file(file_hashes, output_file):
    with open(output_file, 'w') as f:
        for file_hash, file_path in file_hashes.items():
            file_name = file_path.split('/')[-1]  # 获取文件名
            f.write(f"{file_name}:{file_hash}\n")

def parse_hash_file(dir):

    file_path = os.path.join(dir, "hash_seed.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        hash_dict = {}
        for line in lines:
            if ":" in line:
                file_name, file_hash = line.strip().split(':')
                hash_dict[file_name] = file_hash
        return hash_dict
    else:
        return {}

def read_bash_variables(file_path):
    # 正则表达式用于匹配变量赋值，考虑了可能的空格和使用双引号
    pattern = r'^\s*(LDFLAGS|CFLAGS)\s*=\s*"([^"]*)"'
    variables = {'LDFLAGS': None, 'CFLAGS': None}
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(pattern, line)
                if match:
                    variables[match.group(1)] = match.group(2)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except IOError:
        print(f"Error: An error occurred while reading the file {file_path}.")
    
    return (variables["LDFLAGS"],variables["CFLAGS"])

def find_compilers(file_path):
    
    # Open and read the script file
    with open(file_path, 'r') as file:
        script_content = file.read()

    # Check for the presence of g++ and gcc in the content
    if "g++" in script_content:
        return "clang++"

    return "clang"
    
def print_functions(functions):
    """打印函数及其基本块的详细信息，包括 Flatten 级别和 BCF 比例"""
    for functionName, functionInfo in functions.items():
        print(f"Function: {functionName} (Flatten Level: {functionInfo.flatten_level}, BCF Rate: {functionInfo.bcf_rate})")
        for blockNum, blockInfo in functionInfo.blocks.items():
            asm_indices_str = ' '.join(map(str, blockInfo.asm_indices)) if blockInfo.asm_indices else "None"
            print(f"  Block #{blockNum}, Count: {blockInfo.count}, Asm Indices: {asm_indices_str}")

def check_in_folder(directory):
    in_folder = os.path.join(directory, "in")
    # 检查是否存在 in 文件夹
    if os.path.exists(in_folder) and os.path.isdir(in_folder):
        print(f"The folder '{in_folder}' exists.")
        
        # 检查 in 文件夹下是否有文件
        if any(os.path.isfile(os.path.join(in_folder, f)) for f in os.listdir(in_folder)):
            # print(f"The folder '{in_folder}' contains files.")
            return 1
        else:
            # print(f"The folder '{in_folder}' is empty.")
            return 0
    else:
        return 0
        # print(f"The folder '{in_folder}' does not exist.")

 
class Log:
    
    def __init__(self, filename="log"):
        self.log_file = open(f"/root/IRFuzz/{filename}", mode="a")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"Log file created at: {current_time}\n")
    
    def write(self, message):
        self.log_file.write(f"{message}")
        self.log_file.flush()
    
class FuzzLog:
    
    def __init__(self, fuzz_dir, filename="log"):
        self.fuzz_log_file = open(f"{fuzz_dir}/{filename}", mode="w")
    
    def write(self, writestr, color="white"):
        self.fuzz_log_file.write(writestr)
        print(colored(writestr, color))
    
    def flush(self):
        self.fuzz_log_file.flush()

class Fuzz:
    
    def __init__(self, source_dir, fuzz_dir, model, mutator_counts):
        
        self.source_dir = source_dir
        self.fuzz_dir = fuzz_dir
        self.bash_sh = f"{source_dir}/fuzz_compile.sh"
        self.temp_bb_file_path = f"{fuzz_dir}/temp/.basicblock"
        self.model = model
        self.LDFLAGS, self.CFLAGS= read_bash_variables(f"{source_dir}/compile.sh")
        self.compiler = find_compilers(f"{source_dir}/compile.sh")
        self.function_probabilities = {}  # 添加字典以记录函数的概率
        self.iteration = 0
        self.mutator_counts = mutator_counts
        
        self.fuzz_log = FuzzLog(fuzz_dir)
        self.fuzz_log.write(f"Model:{self.model}\n", "blue")
        
        build_fuzz_directories(self.fuzz_dir)
        
        self.bb_file_path =  f"{source_dir}/BasicBlock.txt"
        self.functions = parse_file(self.bb_file_path)
        
        # 示例输出,获取初始概率
        self.fuzz_log.write(f"初始概率为:", "green")
        self.temp_functions = self.functions
        # 将temp输出到temp目录中
        output_file(self.temp_functions, self.temp_bb_file_path)
        self.init_probability_0, self.init_probability_1 = self.get_probability()
        self.adversarial_label = 0 if self.init_probability_0 < self.init_probability_1 else 1 # 哪个概率小，哪个就是对抗样本标签
        self.fuzz_log.write(f"对抗样本label标签为:{self.adversarial_label}\n", "green")
        print(self.init_probability_0, self.init_probability_1)
        # print_functions(self.temp_functions)
    
    def run(self):

        # 随机选择变异器
        mutators = ["random_block", "all_block", "flatten", "bcf"]
        
        copy_file_to_folder(source_file=f"{self.source_dir}/BasicBlock.txt",target_folder=f"{self.fuzz_dir}/in")
        self.file_hashes = parse_hash_file(f"{self.fuzz_dir}/in")
        self.seed_list = [SeedFile(f) for f in list_seed_files(directory=f"{self.fuzz_dir}/in")]
        self.seed_count = len(self.seed_list) - 1
        self.fuzz_log.write(f"there is {self.seed_count} seed files\n","green")
        
        attack_success = False
        iteration = 0
        mutator_counts = { "random_block": 0, "all_block": 0, "flatten": 0, "bcf": 0 }
        
        while not attack_success and iteration < MAX_ITERATIONS:
            # 顺序执行种子
            for seed_file in self.seed_list:
                if iteration >= MAX_ITERATIONS or attack_success:
                    break   
                self.fuzz_log.write(f"Selected seed file: {seed_file.path} with energy {seed_file.energy}\n", "blue")
                # 计算变异次数，基于能量值，能量越高变异次数越多
                num_mutations = max(1, int(seed_file.energy))       # 基础能量就代表变异的次数，至少一次
                functions = parse_file(seed_file.path)              # 解析原函数文件
                copy_functions = copy.deepcopy(functions)           # 保存原有副本
                # 最初对抗标签的概率
                previous_adv_probability = self.init_probability_0 if self.adversarial_label == 0 else self.init_probability_1
                previous_adv_probability_copy = previous_adv_probability
                
                i = 0
                while i <  num_mutations and iteration < MAX_ITERATIONS and not attack_success:
                    mutation_queued = self.seed_count                # 暂时拥有的种子数
                    # chosen_mutator = random.choice(mutators)
                    chosen_mutator = self.choose_mutator_based_on_count()
                    mutator_counts[chosen_mutator] += 1
                    
                    self.fuzz_log.write(f"Chosen mutator: {chosen_mutator}\n", "yellow")
                    functions_copy = copy.deepcopy(functions)       # 保存原有副本
                    
                    if chosen_mutator == "random_block":
                        mutated_function_name = self.mutate_random_block(functions)
                    elif chosen_mutator == "all_block":
                        mutated_function_name = self.mutate_all_block(functions)
                    elif chosen_mutator == "flatten":
                        mutated_function_name = self.mutate_flatten(functions)
                    elif chosen_mutator == "bcf":
                        mutated_function_name = self.mutate_bcf(functions)

                    
                    self.temp_functions = functions                    
                    output_file(self.temp_functions, self.temp_bb_file_path)    # 将temp输出到temp目录中
                    probability_0, probability_1 = self.get_probability()       # 模型预测概率变化
                    adversarial_probability = probability_0 if self.adversarial_label == 0 else probability_1 # 获取对抗样本标签
                    
                    if mutated_function_name != "all" and adversarial_probability - previous_adv_probability_copy > 0:
                        change_in_probability = adversarial_probability - previous_adv_probability_copy
                        self.update_function_probability(mutated_function_name, change_in_probability)
                        
                    # 判断是否保存变异结果
                    if adversarial_probability - previous_adv_probability > 0:  # 概率变化大于5%
                        if adversarial_probability - previous_adv_probability > 0.10:
                            self.mutator_counts[chosen_mutator] += 1
                        previous_adv_probability = adversarial_probability
                        if not is_file_duplicate(seed_order=self.seed_count, fuzz_dir=self.fuzz_dir, file_hashes=self.file_hashes):
                            seed_out_path = f"{self.fuzz_dir}/in/{self.seed_count}_{chosen_mutator}.txt"
                            output_file(functions, seed_out_path)
                            self.fuzz_log.write(f"save functions to {seed_out_path} \n\n")
                            self.seed_count += 1  
                                        
                            new_seed_file = SeedFile(seed_out_path)
                            new_seed_file.update_energy(adversarial_probability)
                            self.seed_list.append(new_seed_file)
                            self.fuzz_log.write(f"New seed file created with energy {new_seed_file.energy}\n", "green")
                        # 判断是否攻击成功，结束循环 当对抗标签的概率超过0.5即攻击成功
                        if adversarial_probability > 0.5:
                            attack_success = True # 攻击成功
                            # 将当前样本的变异策略添加到全局中
                            print("update_mutator_probability_success!")
                            self.update_mutator_probability(mutator_counts)
                            # 把当前攻击成功的种子文件保存起来，后面还要分析
                            seed_out_path = f"{fuzz_dir}/out/success_{self.model}_{self.iteration}.txt"
                            output_file(functions, seed_out_path)
                            self.fuzz_log.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
                            self.fuzz_log.write(f"Now running seedfile: {seed_file.path}\n")
                            self.fuzz_log.write(f"attack susccess mutate_file is {self.seed_count}.txt \n\n")
                            self.fuzz_log.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
                            # ATTACK_SUCCESS_MAP[self.model].append(self.source_dir.split('/')[-1])
                            break
                    else:
                        self.fuzz_log.write(f"Restoring a copy of functions\n", "red")
                        functions = functions_copy # 恢复到这次变异之前
                        
                    if mutation_queued != self.seed_count:          # 如果在这个种子变异的过程中发现了新的种子
                        seed_file.energy = seed_file.energy * 1.5   # 将种子能量变大1.5倍
                        num_mutations = int(seed_file.energy)       # 重新设置变异次数
                        self.fuzz_log.write(f"Updated seed file energy: {seed_file.energy}\n", "red")
                    # else:
                    #     self.fuzz_log.write(f"Restoring a copy of functions\n", "red")
                    #     functions = functions_copy # 恢复到这次变异之前
                    # 因为有的样本会最终停留向在大概40%左右，所以想着后面能不能多给这样的样本一些时间
                    # if adversarial_probability > 0.4:
                    #     continue
                    iteration += 1
                    self.iteration = iteration

        if not attack_success:
            with open(f"{fuzz_dir}/out/failed_{self.model}.txt", "w") as file:
                file.write("1")

        return attack_success
    
    def get_probability(self):
        # 插入+链接
        res = run_bash(script_path= self.bash_sh,
                args=[self.source_dir, self.fuzz_dir, self.temp_bb_file_path, self.LDFLAGS, self.CFLAGS, self.compiler])
        if res == -1:
            print("run fuzz_compile.sh failed! Please check carefully!\n")
            exit()
            
        # 返汇编
        disassemble(fuzz_dir=self.fuzz_dir)
        # 提取cfg
        extract_cfg(fuzz_dir=self.fuzz_dir)
        # 模型预测
        next_state, result, prediction, top_k_indices = measure(self.fuzz_dir, model=self.model) # prediction 0是良性 1是恶意  目前要把恶意转为良性。 result是模型输出的logsoftmax概率
        result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
        formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)
        probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
        probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
        self.fuzz_log.write(f"probability_0 is {probability_0} probability_1:{probability_1} \n\n", "green")

        return probability_0,  probability_1

    def mutate_random_block(self, functions):
        
        if self.iteration >= CHOOSE_FUNCTION_BASED_ON_PROBABILITY * MAX_ITERATIONS:
            self.fuzz_log.write(f"choose_function_based_on_probability\n")
            functionName = self.choose_function_based_on_probability(functions)
        else:
            functionName = random.choice(list(functions.keys()))
        # 选择一个随机块进行操作    
        blockNum = random.choice(list(functions[functionName].blocks.keys()))
        asmIndex = random.randint(0, 26)
        functions[functionName].blocks[blockNum].asm_indices.append(asmIndex)
        self.fuzz_log.write(f"Mutated {functionName} at block {blockNum} with new asmIndex {asmIndex} \n", "magenta")
        return functionName

    def mutate_all_block(self, functions):
        # 对所有块添加同一随机 asmIndex
        asmIndex = random.randint(0, 26)
        for functionName, function in functions.items():
            for blockNum in function.blocks:
                functions[functionName].blocks[blockNum].asm_indices.append(asmIndex)
        self.fuzz_log.write(f"Mutated all blocks with asmIndex {str(asmIndex)} \n", "magenta")
        return "all"

    def mutate_flatten(self, functions):
        # 随机选择函数并增加 flatten 次数
        if self.iteration >= CHOOSE_FUNCTION_BASED_ON_PROBABILITY * MAX_ITERATIONS:
            self.fuzz_log.write(f"choose_function_based_on_probability\n")
            functionName = self.choose_function_based_on_probability(functions)
        else:
            functionName = random.choice(list(functions.keys()))
        if functions[functionName].flatten_level == 3:
            pass
        else:
            functions[functionName].flatten_level += 1
        self.fuzz_log.write(f"Increased flatten level for {functionName} to {functions[functionName].flatten_level}\n", "magenta")
        return functionName
    
    def mutate_bcf(self, functions):
        
        if self.iteration >= CHOOSE_FUNCTION_BASED_ON_PROBABILITY * MAX_ITERATIONS:
            self.fuzz_log.write(f"choose_function_based_on_probability\n")
            functionName = self.choose_function_based_on_probability(functions)
        else:
            functionName = random.choice(list(functions.keys()))
        # 随机选择函数并增加 bcf 概率
        functionName = random.choice(list(functions.keys()))
        functions[functionName].bcf_rate += 10
        self.fuzz_log.write(f"Increased bcf rate for {functionName} to {functions[functionName].bcf_rate}\n", "magenta")
        return functionName

    def update_function_probability(self, function_name, change_amount):
        if function_name not in self.function_probabilities:
            self.function_probabilities[function_name] = 1
        self.function_probabilities[function_name] += change_amount  # 根据变化幅度调整选择概率

    def choose_function_based_on_probability(self, functions):
        function_names = list(functions.keys())
        probabilities = [self.function_probabilities.get(fn, 1) for fn in function_names]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        return random.choices(function_names, probabilities)[0]
    
    def update_mutator_probability(self, mutator_counts):
        for key in mutator_counts.keys():
            self.mutator_counts[key] += mutator_counts[key]
            
    def update_mutator_probability_fail(self, mutator_counts):
        for key in mutator_counts.keys():
            self.mutator_counts[key] -= mutator_counts[key]
            if self.mutator_counts[key] < 0: 
                self.mutator_counts[key] = 0
            
    def choose_mutator_based_on_count(self):
        probabilities = [(self.mutator_counts.get(mutator, 0) + 1) for mutator in self.mutator_counts]
        total = sum(probabilities)
        probabilities = [prob / total for prob in probabilities]
        return random.choices(list(self.mutator_counts.keys()), probabilities)[0]
    
def is_success_file_present(fuzz_dir, model):
    success_files = glob.glob(f"{fuzz_dir}/out/success_{model}*")
    return len(success_files) > 0     
    
def load_mutator_counts(filepath):
    try:
        print(f"try open file {filepath}")
        with open(filepath, 'r') as file:
            mutator_counts = json.load(file)
            return mutator_counts
    except FileNotFoundError:
        mutator_counts = { "random_block": 0, "all_block": 0, "flatten": 0, "bcf": 0 }
        return mutator_counts
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}. Using default mutator counts.")
        
if __name__ == "__main__":

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    parser.add_argument('--model_list', type=str, nargs='+', default=["DGCNN_9","DGCNN_20","GIN0_9","GIN0_20","GIN0WithJK_9","GIN0WithJK_20"],
                        help='List of models to use')
    parser.add_argument('--max_iterations', type=int, default=30,
                        help='Maximum number of iterations')
    parser.add_argument('--choose_function_based_on_probability', type=float, default=0.8,
                        help='Probability for choosing function in the last 20% phase')
    parser.add_argument('--malware_store_path', type=str, default="/root/IRFuzz/ELF",
                        help='Path to the malware store')

    args = parser.parse_args()
    
    model_list = args.model_list
    malware_store_path = args.malware_store_path
    MAX_ITERATIONS = args.max_iterations                                                # 最大迭代次数
    CHOOSE_FUNCTION_BASED_ON_PROBABILITY = args.choose_function_based_on_probability    # 在最后的20%阶段,按照概率变化来选择函数
    ATTACK_SUCCESS_MAP = {model: [] for model in model_list}
    ATTACK_SUCCESS_RATE = dict()    
    LOGFILE = Log()                                                                     # 全局的日志文件
    
    LOGFILE.write(f"MAX_ITERATIONS:{MAX_ITERATIONS}\n CHOOSE_FUNCTION_BASED_ON_PROBABILITY:{CHOOSE_FUNCTION_BASED_ON_PROBABILITY}\n model_list:{model_list}\n\n")
    malware_full_paths = [os.path.join(malware_store_path, entry) for entry in os.listdir(malware_store_path)]
    
    total_iterations = len(model_list) * len(malware_full_paths)
    progressed = 0    

    for model in model_list:
        
        mutator_counts = load_mutator_counts(f"/root/IRFuzz/attack_success_mutation_{model}.json")
                
        for malware_dir in malware_full_paths:
            source_dir= malware_dir
            fuzz_dir=  malware_dir
            model = model
            print(f"{mutator_counts}")
            print("Now is process {:.2f}%".format( (progressed/total_iterations)*100 ))
            if is_success_file_present(fuzz_dir,model):
                print(colored(f"Already Attack Success! Next One!", "green"))
                ATTACK_SUCCESS_MAP[model].append(source_dir.split('/')[-1])
                progressed += 1
                continue
            if os.path.exists(f"{fuzz_dir}/out/failed_{model}.txt"):
                print(colored(f"Already Failed!", "yellow"))
                progressed += 1
                continue
            
            startime =  datetime.datetime.now()
            
            fuzz = Fuzz(source_dir,fuzz_dir,model,mutator_counts)
            attack_success = fuzz.run()
            mutator_counts = fuzz.mutator_counts
            with open('/root/IRFuzz/attack_success_mutation.json', 'w') as file:
                json.dump(mutator_counts, file)
            endtime =  datetime.datetime.now()
            if attack_success:
                LOGFILE.write(f"{model}-{source_dir.split('/')[-1]}\n")
                ATTACK_SUCCESS_MAP[model].append(source_dir.split('/')[-1])
                LOGFILE.write(f"Use {(endtime - startime).total_seconds()} s\n\n")

            progressed += 1
    
    for key in ATTACK_SUCCESS_MAP:
        ATTACK_SUCCESS_RATE[key] = len(ATTACK_SUCCESS_MAP[key]) / len(malware_full_paths)
    
    with open('/root/IRFuzz/attack_success_object.txt', 'w') as file:
        for key, object in ATTACK_SUCCESS_MAP.items():
            file.write(f'{key}: {str(object)}\n')  # 输出格式化的浮点数   
    
    with open('/root/IRFuzz/attack_success_rate.txt', 'w') as file:
        for key, value in ATTACK_SUCCESS_RATE.items():
            file.write(f'{key}: {value:.4f}\n')  # 输出格式化的浮点数
    
    exit()