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


def parse_asm_indices(indices_str):
    """解析汇编指令索引字符串，返回索引列表"""
    return [int(idx) for idx in indices_str.split('+') if idx.strip()]

def parse_file(file_path):
    """解析文件并填充包含多个函数信息的字典，现在包括Flatten层级和BCF比例"""
    functions = {}
    current_function = None

    with open(file_path, 'r') as file:
        for line in file:
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
    result = run_measure.measure(data_dir=data_dir, model_name=model)
    return result

def add_random_asmIndex(functions, changes_log):
    # 随机挑选一个函数名
    max_log_prob = 0.8
    log_len = len(changes_log)
    # 计算从changes_log中挑选的概率，但不超过max_log_prob = 0.8
    prob = min(log_len / (log_len + 100), max_log_prob)  # 示例逻辑，可根据需要调整

    if random.random() < prob and  changes_log is not []:
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

def add_random_asmIndex_to_all_blocks(functions, changes_log, discarded_changes_log):
    
    valid_range = set(range(27))  # 创建一个从0到26的集合
    valid_numbers = list(valid_range - set(discarded_changes_log)- set(changes_log))  # 移除被排除的数字，暂时也把changes_log的也移除
    asmIndex = random.choice(valid_numbers)  # 随机选择一个有效的数字
    return_asmIndex = {"asmIndex": asmIndex}
    # 对每个函数进行遍历
    for functionName, function in functions.items():
        # 对该函数下的每个基本块进行遍历
        for blockNum, block in function.blocks.items():
            # 将asmIndex添加到当前遍历的基本块的asm_indices列表中
            block.asm_indices.append(asmIndex)
            print(f"Added - ALL Function ALL Block asmIndex: {asmIndex}")
    
    return return_asmIndex, asmIndex

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
        self.temp_functions = self.functions

        build_fuzz_directories(self.fuzz_dir)
        # copy_file_to_folder(source_file=f"{self.source_dir}/BasicBlock.txt",target_folder=f"{self.fuzz_dir}/out")
        # self.file_hashes = parse_hash_file(f"{self.fuzz_dir}/out")
        # self.seed_list = list_seed_files(directory=f"{self.fuzz_dir}/out")
        
        # 示例输出,获取初始概率
        self.fuzz_log.write(f"初始概率为:", "green")
        self.init_probability_0, self.init_probability_1 = self.get_probability()
        print(self.init_probability_0, self.init_probability_1)
        print_functions(self.temp_functions)
    
    def run(self, mutate):
        
        if mutate == "ALL BLOCK":
            copy_file_to_folder(source_file=f"{self.source_dir}/BasicBlock.txt",target_folder=f"{self.fuzz_dir}/out/all_block")
            self.file_hashes = parse_hash_file(f"{self.fuzz_dir}/out/all_block")
            self.seed_list = list_seed_files(directory=f"{self.fuzz_dir}/out/all_block")
            self.seed_count = len(self.seed_list) - 1
            self.fuzz_log.write(f"there is {self.seed_count} seed files","green")
            self.run_all_block()
            
        elif mutate == "RANDOM BLOCK":
            copy_file_to_folder(source_file=f"{self.source_dir}/BasicBlock.txt",target_folder=f"{self.fuzz_dir}/out/random_block")
            self.file_hashes = parse_hash_file(f"{self.fuzz_dir}/out/random_block")
            self.seed_list = list_seed_files(directory=f"{self.fuzz_dir}/out/random_block")
            self.seed_count = len(self.seed_list) - 1
            self.fuzz_log.write(f"there is {self.seed_count} seed files","green")
            self.run_random_block()
        elif mutate == "flatten":

            pass
        
        elif mutate == "bogus":
            
            pass 
        else:
            raise NotImplementedError
    
    def run_all_block(self):
        self.fuzz_log.write(f"**All block mutator** \n","red")
        previous_probability_0 = self.init_probability_0
        # self.seed_count = 0
        attack_success = False
        selected_files = []

        while not attack_success:
            # 从未选择过的文件中优先选择
            choices = [file for file in self.seed_list if file not in selected_files]
            # 如果所有文件都已选择过，重置选择列表以重新开始
            if not choices:
                selected_files = []
                choices = self.seed_list[:]
                self.fuzz_log.write(f"攻击失败！^@^\n")
                exit()
            basicblockfile = random.choice(choices)
            # 将选择的文件添加到已选择列表中
            selected_files.append(basicblockfile)   
            
            self.fuzz_log.write(f"Now is {basicblockfile}\n")
            num_episodes = 27 # 一个文件，我们变异27次
            functions = parse_file(basicblockfile) # 解析文件
            copy_functions = copy.deepcopy(functions) # 保存原有副本
            changes_log = [] # 保留之前让文件良性概率增加的操作
            discarded_changes_log = [] # 抛弃那些添加操作之后，反而使良性概率降低的操作
            while num_episodes > 0 and not attack_success:
                self.fuzz_log.write("******************")
                self.fuzz_log.write(f"num_episodes is {num_episodes}\n")
                self.fuzz_log.write(f"changes_log is {changes_log}\n")
                self.fuzz_log.write(f"discarded_changes_log is {discarded_changes_log}\n")
                
                num_episodes -= 1
                if len(discarded_changes_log) == 26:
                    num_episodes = -1 # 终止这次循环
                    self.fuzz_log.write(f"Terminate this mutation because there is no operation that can improve the probability", 'red')
                    continue
                
                change,asmIndex = add_random_asmIndex_to_all_blocks(functions, changes_log, discarded_changes_log) #增加随机变异,用change保留下来
                self.fuzz_log.write(f"chang is {change}\n")
                # 将变化后的functions输出到self.temp_functions
                self.temp_functions = functions
                probability_0, probability_1 = self.get_probability()
                
                # 如果良性的概率增加,将当前变异策略输出到out目录当中
                if  probability_0 > previous_probability_0:
                    # 检查文件是否重复
                    if not is_file_duplicate(seed_order=self.seed_count, fuzz_dir=self.fuzz_dir, file_hashes=self.file_hashes):
                        output_file(functions, f"{self.fuzz_dir}/out/all_block/{self.seed_count}.txt")
                        self.fuzz_log.write(f"save functions to {self.fuzz_dir}/out/all_block/{self.seed_count}.txt \n\n")
                        self.seed_count += 1
                    else:
                        self.fuzz_log.write(f"the seed file is duplicate","green")
                        
                    # 如果返回的操作不为空，将其保存在changes_log中
                    changes_log.append(asmIndex)
                        
                # 如果没有这种变化,就舍弃当前给出的随机变化,重新将functions设置为原始
                # 此外，保存这个change，从此不再使用这个变化
                # TODO: discarded_changes_log 目前只在add_random_asmIndex_to_all_blocks中发挥作用
                else:
                    # 如果没有这种变化,就舍弃当前给出的随机变化,重新将functions设置为原始
                    # functions= copy.deepcopy(copy_functions) 
                    # 每次都设置成最开始进入循环的样子
                    discarded_changes_log.append(asmIndex) # 抛弃那些添加操作之后，反而使良性概率降低的操作
                    
                # functions= copy.deepcopy(copy_functions)
                
                if probability_0 > probability_1:
                    attack_success = True
                    self.fuzz_log.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
                    self.fuzz_log.write(f"Now running basicblockfile: {basicblockfile}\n")
                    self.fuzz_log.write(f"attack susccess functions is {self.seed_count}.txt \n\n")
                    self.fuzz_log.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
                    
                    
                
            self.fuzz_log.flush()  # 强制将缓冲区内容写入文件，但不关闭文件

            # 重新加载文件夹中的种子内容
            self.seed_list = list_seed_files(directory=f"{self.fuzz_dir}/all_block")
            
    def run_random_block(self):
        self.fuzz_log.write(f"**Random block mutator** \n","red")
        previous_probability_0 = self.init_probability_0
        # self.seed_count = 0
        attack_success = False
        selected_files = []
        
        while not attack_success:
            # 从未选择过的文件中优先选择
            choices = [file for file in self.seed_list if file not in selected_files]
            # 如果所有文件都已选择过，重置选择列表以重新开始
            if not choices:
                selected_files = []
                choices = self.seed_list[:]
                # self.fuzz_log.write(f"攻击失败！^@^\n")
                # exit()
            basicblockfile = random.choice(choices)
            # 将选择的文件添加到已选择列表中
            selected_files.append(basicblockfile)   
            
            self.fuzz_log.write(f"Now is {basicblockfile}\n")
            num_episodes = 27 # 一个文件，我们变异26次
            functions = parse_file(basicblockfile) # 解析文件
            copy_functions = copy.deepcopy(functions) # 保存原有副本
            changes_log = [] # 保留之前让文件良性概率增加的操作
            discarded_changes_log = [] # 抛弃那些添加操作之后，反而使良性概率降低的操作
            while num_episodes > 0 and not attack_success :
                num_episodes -= 1
                self.fuzz_log.write("******************")
                self.fuzz_log.write(f"num_episodes is {num_episodes}\n")
                self.fuzz_log.write(f"changes_log is {changes_log}\n")
                
                previous_functions = copy.deepcopy(functions) # 保存上一次的副本
                change = add_random_asmIndex(functions, changes_log) #增加随机变异,用change保留下来
                self.fuzz_log.write(f"chang is {change}\n")
                # 将变化后的functions输出到self.temp_functions
                self.temp_functions = functions
                probability_0, probability_1 = self.get_probability()
                
                
                # 如果良性的概率增加,将当前变异策略输出到out目录当中
                # 最开始为提高随机性，只要大于0都保存
                if  probability_0 > previous_probability_0:
                    previous_probability_0 = probability_0
                    # 检查文件是否重复
                    if not is_file_duplicate(seed_order=self.seed_count, fuzz_dir=self.fuzz_dir, file_hashes=self.file_hashes):
                        output_file(functions, f"{self.fuzz_dir}/out/random_block/{self.seed_count}.txt")
                        self.fuzz_log.write(f"save functions to {self.fuzz_dir}/out/random_block/{self.seed_count}.txt \n\n")
                        self.seed_count += 1
                    else:
                        self.fuzz_log.write(f"the seed file is duplicate","green")
                        
                    # 如果返回的操作不为空，将其保存在changes_log中
                    if  len(change) != 0 : 
                        changes_log.append(change)
                        
                # 如果没有这种变化,就舍弃当前给出的随机变化,重新将functions设置为原始
                # 此外，保存这个change，从此不再使用这个变化
                # else:
                    # functions= copy.deepcopy(copy_functions) 
                    # discarded_changes_log.append(asmIndex) # 抛弃那些添加操作之后，反而使良性概率降低的操作
                    # discarded_changes_log.append(change) # 抛弃那些添加操作之后，反而使良性概率降低的操作
                # TODO: 这个地方可以在商榷一下，是广度还是深度优先，也就是到底是在变异操作应用的基础上继续变异，
                # 还是说，回到变异操作应用之前，去尝试更多的变异操作
                else:
                    # functions= copy.deepcopy(copy_functions) 
                    # functions = previous_functions
                    pass
                
                if probability_0 > probability_1:
                    attack_success = True
                    self.fuzz_log.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
                    self.fuzz_log.write(f"Now running basicblockfile: {basicblockfile}\n")
                    self.fuzz_log.write(f"attack susccess functions is {self.seed_count}.txt \n\n")
                    self.fuzz_log.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
                
            self.fuzz_log.flush()  # 强制将缓冲区内容写入文件，但不关闭文件

            # 重新加载文件夹中的种子内容
            self.seed_list = list_seed_files(directory=f"{self.fuzz_dir}/out/random_block")
                       
    def get_probability(self):
        
        # 将temp输出到temp目录中
        output_file(self.temp_functions, self.temp_bb_file_path)
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

     

if __name__ == "__main__":
    # source_dir="/home/lebron/disassemble/attack/sourcecode/Linux.Phide/attack"
    # fuzz_dir="/home/lebron/IRFuzz/Linux.Phide"
    # bash_sh = "/home/lebron/disassemble/attack/sourcecode/Linux.Phide/attack/fuzz_insertasminstruction.sh"
    
    # /home/lebron/IRFuzz/Linux.Apachebd
    source_dir="/home/lebron/disassemble/attack/sourcecode/Linux.Apachebd/attack"
    fuzz_dir="/home/lebron/IRFuzz/Linux.Apachebd"
    bash_sh = "/home/lebron/disassemble/attack/sourcecode/Linux.Apachebd/attack/fuzz_insertasminstruction.sh"
    # temp_bb_file_path = f"{fuzz_dir}/temp/.basicblock"
    model = "semantics_dgcnn"   # dgcnn  semantics_dgcnn
    fuzz = Fuzz(source_dir,fuzz_dir,bash_sh,model)
    # fuzz.run(mutate="ALL BLOCK")
    fuzz.run(mutate="RANDOM BLOCK")
    
    exit()




def main():
    
    # source_dir="/home/lebron/disassemble/attack/sourcecode/Linux.Apachebd/attack"
    source_dir="/home/lebron/disassemble/attack/sourcecode/Linux.Phide/attack"
    fuzz_dir="/home/lebron/IRFuzz/Linux.Phide"
    bash_sh = "/home/lebron/disassemble/attack/sourcecode/Linux.Phide/attack/fuzz_insertasminstruction.sh"
    temp_bb_file_path = f"{fuzz_dir}/temp/.basicblock"
    model = "semantics_dgcnn"   # dgcnn
    # semantics_dgcnn 是论文中的 9维基本块特征, 每个维度是基本块指令出现的次数
    # dgcnn  是20维特征
    
    file_path = f"{source_dir}/BasicBlock.txt"
    functions = parse_file(file_path)
    fuzz_log = FuzzLog(fuzz_dir)
    # 示例输出
    for functionName, functionInfo in functions.items():
        print(f"Function: {functionName}")
        for blockNum, blockInfo in functionInfo.blocks.items():
            print(f"  Block #{blockNum}, Count: {blockInfo.count}, Asm Indices: {' '.join(map(str, blockInfo.asm_indices))}")

    
    build_fuzz_directories(fuzz_dir)
    copy_file_to_folder(source_file=f"{source_dir}/BasicBlock.txt",target_folder=f"{fuzz_dir}/out")
    file_hashes = parse_hash_file(f"{fuzz_dir}/out")
    seed_list = list_seed_files(directory=f"{fuzz_dir}/out")
    
    previous_probability_0 = 0
    seed_count = 0
    attack_success = False
    selected_files = []
    # for basicblockfile in seed_list:
    while not attack_success:
        # 从未选择过的文件中优先选择
        choices = [file for file in seed_list if file not in selected_files]
        # 如果所有文件都已选择过，重置选择列表以重新开始
        if not choices:
            selected_files = []
            choices = seed_list[:]
            fuzz_log.write(f"攻击失败！^@^\n")
            exit()
        basicblockfile = random.choice(choices)
        # 将选择的文件添加到已选择列表中
        selected_files.append(basicblockfile)
        
        fuzz_log.write(f"Now is {basicblockfile}\n")
        num_episodes = 27 # 一个文件，我们变异26次，第一次会减去一次
        functions = parse_file(basicblockfile) # 解析文件
        copy_functions = copy.deepcopy(functions) # 保存原有副本
        changes_log = [] # 保留之前让文件良性概率增加的操作
        discarded_changes_log = [] # 抛弃那些添加操作之后，反而使良性概率降低的操作
        while num_episodes >= 0:
            fuzz_log.write("******************")
            fuzz_log.write(f"num_episodes is {num_episodes}\n")
            fuzz_log.write(f"changes_log is {changes_log}\n")
            fuzz_log.write(f"discarded_changes_log is {discarded_changes_log}\n")
            
            num_episodes -= 1
            # change = add_random_asmIndex(functions,changes_log) #增加随机变异,用change保留下来
            # if len(discarded_changes_log) == 26:
            #     num_episodes = -1 # 终止这次循环
            #     fuzz_log.write(f"Terminate this mutation because there is no operation that can improve the probability", 'red')
            #     continue
            
            # change,asmIndex = add_random_asmIndex_to_all_blocks(functions, changes_log, discarded_changes_log) #增加随机变异,用change保留下来
            change = add_random_asmIndex(functions, changes_log) #增加随机变异,用change保留下来
            fuzz_log.write(f"chang is {change}\n")

            output_file(functions, temp_bb_file_path)
            # 插入+链接
            run_bash(script_path= bash_sh,
                    args=[source_dir, fuzz_dir, temp_bb_file_path])
            # 返汇编
            disassemble(fuzz_dir=fuzz_dir)
            # 提取cfg
            extract_cfg(fuzz_dir=fuzz_dir)
            # 模型预测
            next_state, result, prediction, top_k_indices = measure(fuzz_dir, model=model) # prediction 0是良性 1是恶意  目前要把恶意转为良性。 result是模型输出的logsoftmax概率
            result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
            formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)

            probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
            probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
            fuzz_log.write(f"probability_0 is {probability_0} probability_1:{probability_1} \n\n ", "green")
            # 第一次进来,先不变异.为了获取初始恶意类别的分类概率
            if num_episodes != 26:
                # 如果良性的概率增加,将当前变异策略输出到out目录当中
                if  probability_0 > previous_probability_0: 
                    # 刚开始只要比最初始的概率大，就保存。
                    # 当changes_logs到达50个的时候，那么提升不大的操作就不保存。
                    if len(changes_log) > 50:
                        previous_probability_0 = probability_0
                    # 检查文件是否重复
                    if not is_file_duplicate(seed_order=seed_count, fuzz_dir=fuzz_dir, file_hashes=file_hashes):
                        output_file(functions, f"{fuzz_dir}/out/{seed_count}.txt")
                        fuzz_log.write(f"save functions to {fuzz_dir}/out/{seed_count}.txt \n")
                        
                        seed_count += 1
                    else:
                        fuzz_log.write(f"the seed file is duplicate","green")
                        
                    # 如果返回的操作不为空，将其保存在changes_log中
                    if not change : 
                        changes_log.append(change)
                        # changes_log.append(asmIndex)
                        
                # 如果没有这种变化,就舍弃当前给出的随机变化,重新将functions设置为原始
                # 此外，保存这个change，从此不再使用这个变化
                # TODO: discarded_changes_log 目前只在add_random_asmIndex_to_all_blocks中发挥作用
                else:
                    functions= copy.deepcopy(copy_functions) 
                    # discarded_changes_log.append(asmIndex) # 抛弃那些添加操作之后，反而使良性概率降低的操作
                    discarded_changes_log.append(change) # 抛弃那些添加操作之后，反而使良性概率降低的操作
                    
            
            if probability_0 > probability_1:
                attack_success = True
                print(f"Now running basicblockfile: {basicblockfile}")
            
        fuzz_log.flush()  # 强制将缓冲区内容写入文件，但不关闭文件

        # 重新加载文件夹中的种子内容
        seed_list = list_seed_files(directory=f"{fuzz_dir}/out")

       