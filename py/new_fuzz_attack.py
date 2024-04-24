import os
import random
import shutil
import copy
import torch
from termcolor import colored

from model.src.dataset_construct import run_disassemble
from model.src.dataset_construct import run_extract_cfg
from model.src.gnn import run_measure
import hashlib

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
        self.energy = 10
        self.asm_insert_count = 0
        self.flatten_count = 0
        self.bcf_count = 0
        self.load_and_parse()

    def update_energy(self, probability_adversarial_label, iteration):
        # 基于概率接近0.5的程度，以及变异操作次数计算能量
        distance = abs(0.5 - probability_adversarial_label)
        base_energy = (0.5 - distance) * 100  # 距离越近，能量越大
        time_factor = max(0.1, 1 - (iteration / 1000))  # 随时间减少分配给距离较远的种子的能量
        self.energy = base_energy * time_factor - (self.asm_insert_count + self.flatten_count * 2 + self.bcf_count * 2)
            
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
    folders = ["in", "out", "out/random_block", "out/all_block" ,"rawdata", "asm", "cfg", "cfg_magic_test","temp"]

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
    result = run_measure.measure(data_dir=data_dir, model=model)
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
    store_dir = f"{fuzz_dir}/out"
    
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

def print_functions(functions):
    """打印函数及其基本块的详细信息，包括 Flatten 级别和 BCF 比例"""
    for functionName, functionInfo in functions.items():
        print(f"Function: {functionName} (Flatten Level: {functionInfo.flatten_level}, BCF Rate: {functionInfo.bcf_rate})")
        for blockNum, blockInfo in functionInfo.blocks.items():
            asm_indices_str = ' '.join(map(str, blockInfo.asm_indices)) if blockInfo.asm_indices else "None"
            print(f"  Block #{blockNum}, Count: {blockInfo.count}, Asm Indices: {asm_indices_str}")

    
class FuzzLog:
    
    def __init__(self, fuzz_dir):
        self.fuzz_log_file = open(f"{fuzz_dir}/log", mode="w")
    
    def write(self, writestr, color="white"):
        self.fuzz_log_file.write(writestr)
        print(colored(writestr, color))
    
    def flush(self):
        self.fuzz_log_file.flush()

class Fuzz:
    
    def __init__(self, source_dir, fuzz_dir, bash_sh, model):
        
        self.source_dir = source_dir
        self.fuzz_dir = fuzz_dir
        self.bash_sh = f"{source_dir}/fuzz_insertasminstruction.sh"
        self.temp_bb_file_path = f"{fuzz_dir}/temp/.basicblock"
        self.model = model
        
        self.bb_file_path =  f"{source_dir}/BasicBlock.txt"
        self.fuzz_log = FuzzLog(fuzz_dir)
        self.functions = parse_file(self.bb_file_path)

        build_fuzz_directories(self.fuzz_dir)
        
        # 示例输出,获取初始概率
        self.fuzz_log.write(f"初始概率为:", "green")
        self.temp_functions = self.functions
        # 将temp输出到temp目录中
        output_file(self.temp_functions, self.temp_bb_file_path)
        self.init_probability_0, self.init_probability_1 = self.get_probability()
        self.adversarial_label = 0 if self.init_probability_0 < self.init_probability_1 else 1 # 哪个概率小，哪个就是对抗样本标签
        self.fuzz_log.write(f"对抗样本label标签为:{self.adversarial_label}", "green")
        print(self.init_probability_0, self.init_probability_1)
        print_functions(self.temp_functions)
    
    
    def run(self):
        # 最大迭代次数
        MAX_ITERATIONS=1000
        # 随机选择变异器
        # mutators = ["random_block", "all_block", "flatten", "bcf"]
        mutators = ["random_block"]
        
        
        iteration = 0
        copy_file_to_folder(source_file=f"{self.source_dir}/BasicBlock.txt",target_folder=f"{self.fuzz_dir}/out")
        self.file_hashes = parse_hash_file(f"{self.fuzz_dir}/out")
        self.seed_list = [SeedFile(f) for f in list_seed_files(directory=f"{self.fuzz_dir}/out")]
        self.seed_count = len(self.seed_list) - 1
        self.fuzz_log.write(f"there is {self.seed_count} seed files","green")
        
        while True:
            # 顺序执行种子
            for seed_file in self.seed_list:
                
                self.fuzz_log.write(f"Selected seed file: {seed_file.path} with energy {seed_file.energy}", "blue")
                # 计算变异次数，基于能量值，能量越高变异次数越多
                num_mutations = max(1, int(seed_file.energy))       # 基础能量就代表变异的次数，至少一次
                functions = parse_file(seed_file.path)              # 解析原函数文件
                copy_functions = copy.deepcopy(functions)           # 保存原有副本
                # 最初对抗标签的概率
                previous_adv_probability = self.init_probability_0 if self.adversarial_label == 0 else self.init_probability_1 
                mutation_queued = self.seed_count # 暂时拥有的种子数
                
                # for _ in range(num_mutations):
                i = 0
                while i <  num_mutations:
                    chosen_mutator = random.choice(mutators)
                    self.fuzz_log.write(f"Chosen mutator: {chosen_mutator}", "yellow")

                    if chosen_mutator == "random_block":
                        self.mutate_random_block(functions)
                    elif chosen_mutator == "all_block":
                        self.mutate_all_block(functions)
                    elif chosen_mutator == "flatten":
                        self.mutate_flatten(functions)
                    elif chosen_mutator == "bcf":
                        self.mutate_bcf(functions)

                    
                    self.temp_functions = functions                    
                    output_file(self.temp_functions, self.temp_bb_file_path)    # 将temp输出到temp目录中
                    probability_0, probability_1 = self.get_probability()       # 模型预测概率变化
                    adversarial_probability = probability_0 if self.adversarial_label == 0 else probability_1 # 获取对抗样本标签
                    
                    # 判断是否保存变异结果
                    if abs(adversarial_probability - previous_adv_probability) > 0.05:  # 概率变化大于5%
                        if not is_file_duplicate(seed_order=self.seed_count, fuzz_dir=self.fuzz_dir, file_hashes=self.file_hashes):
                            seed_out_path = f"{self.fuzz_dir}/out/{self.seed_count}_{chosen_mutator}.txt"
                            output_file(functions, seed_out_path)
                            self.fuzz_log.write(f"save functions to {seed_out_path} \n\n")
                            self.seed_count += 1  
                                        
                            new_seed_file = SeedFile(seed_out_path)
                            new_seed_file.update_energy(adversarial_probability, iteration)
                            self.seed_list.append(new_seed_file)
                            self.fuzz_log.write(f"New seed file created with energy {new_seed_file.energy}", "green")
                        # 判断是否攻击成功，结束循环 当对抗标签的概率超过0.5即攻击成功
                        if adversarial_probability > 0.5:
                            attack_success = True
                            self.fuzz_log.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
                            self.fuzz_log.write(f"Now running seedfile: {seed_file.path}\n")
                            self.fuzz_log.write(f"attack susccess mutate_file is {self.seed_count}.txt \n\n")
                            self.fuzz_log.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
                            exit()
                    if mutation_queued != self.seed_count:          # 如果在这个种子变异的过程中发现了新的种子
                        seed_file.energy = seed_file.energy * 1.5   # 将种子能量变大1.5倍
                        num_mutations = int(seed_file.energy)       # 重新设置变异次数
                        self.fuzz_log.write(f"Updated seed file energy: {seed_file.energy}", "red")

            iteration += 1
            if iteration >= MAX_ITERATIONS:
                break
                    
    def get_probability(self):
        # 插入+链接
        run_bash(script_path= self.bash_sh,
                args=[self.source_dir, self.fuzz_dir, self.temp_bb_file_path])
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
        self.fuzz_log.write(f"probability_0 is {probability_0} probability_1:{probability_1} \n", "green")

        return probability_0,  probability_1

    def mutate_random_block(self, functions):
        # 选择一个随机块进行操作
        functionName = random.choice(list(functions.keys()))
        blockNum = random.choice(list(functions[functionName].blocks.keys()))
        asmIndex = random.randint(0, 26)
        functions[functionName].blocks[blockNum].asm_indices.append(asmIndex)
        self.fuzz_log.write(f"Mutated {functionName} at block {blockNum} with new asmIndex {asmIndex}", "magenta")

    def mutate_all_block(self, functions):
        # 对所有块添加同一随机 asmIndex
        asmIndex = random.randint(0, 26)
        for functionName, function in functions.items():
            for blockNum in function.blocks:
                functions[functionName].blocks[blockNum].asm_indices.append(asmIndex)
        self.fuzz_log.write("Mutated all blocks with asmIndex " + str(asmIndex), "magenta")

    def mutate_flatten(self, functions):
        # 随机选择函数并增加 flatten 次数
        functionName = random.choice(list(functions.keys()))
        functions[functionName].flatten_level += 1
        self.fuzz_log.write(f"Increased flatten level for {functionName} to {functions[functionName].flatten_level}", "magenta")

    def mutate_bcf(self, functions):
        # 随机选择函数并增加 bcf 概率
        functionName = random.choice(list(functions.keys()))
        functions[functionName].bcf_rate += 10
        self.fuzz_log.write(f"Increased bcf rate for {functionName} to {functions[functionName].bcf_rate}", "magenta")

     

if __name__ == "__main__":

    source_dir="/home/lebron/disassemble/attack/sourcecode/Linux.Apachebd/attack"
    fuzz_dir="/home/lebron/IRFuzz/Linux.Apachebd"
    bash_sh = "/home/lebron/disassemble/attack/sourcecode/Linux.Apachebd/attack/fuzz_insertasminstruction.sh"
    model = "semantics_dgcnn"   # dgcnn  semantics_dgcnn
    fuzz = Fuzz(source_dir,fuzz_dir,bash_sh,model)
    fuzz.run()
    exit()