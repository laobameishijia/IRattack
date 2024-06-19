import copy
import datetime
import glob
import json
import os
import random
import re
import shutil
import subprocess
import time
from termcolor import colored
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
import numpy as np
from collections import deque


from model.src.dataset_construct import run_disassemble
from model.src.dataset_construct import run_extract_cfg
from model.src.gnn import run_measure

from model.src.gnn.model import GIN0WithJK, GIN0, DGCNN
from model.src.gnn.dataset import CFGDataset_Semantics_Preseving, CFGDataset_MAGIC,CFGDataset_MAGIC_Attack
from torch_geometric.loader import DataLoader

class BlockInfo:
    def __init__(self, count, asm_indices):
        self.count = count
        self.asm_indices = asm_indices

class FunctionInfo:
    def __init__(self, flatten_level=0, bcf_rate=0):
        self.flatten_level = flatten_level # 平坦化多少次
        self.bcf_rate = bcf_rate # 基本块作虚拟控制流的概率
        self.blocks = {}


# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)



class Log:
    
    def __init__(self, fuzz_dir="/home/lebron/IRFuzz" , filename="SRL"):
        self.log_file = open(f"{fuzz_dir}/{filename}", mode="a")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"Log file created at: {current_time}\n")
    
    def write(self, message, color="white"):
        self.log_file.write(f"{message}")
        print(colored(message, color))
        self.log_file.flush()


# 计算状态差异
def Diff(s1, s2):
    return np.linalg.norm(s1 - s2)

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
    shutil.copy("/home/lebron/IRFuzz/bash/basicblockcount.sh", fuzz_dir)
    shutil.copy("/home/lebron/IRFuzz/bash/fuzz_compile.sh", fuzz_dir)
    # 运行basicblockcount.sh 输出basicblock.txt
    run_bash(f"{fuzz_dir}/basicblockcount.sh", args=[])

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
    
def select_random_blocks(basic_block_count, num_select=30):
    blocks = list(range(basic_block_count))
    if basic_block_count <= num_select:
        return blocks
    else:
        return random.sample(blocks, num_select)
    
class SRL():
    
    def __init__(self, fuzz_dir, model):

        self.fuzz_dir = fuzz_dir
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.temp_bb_file_path = f"{fuzz_dir}/temp/.basicblock"
        self.bash_sh = f"{fuzz_dir}/fuzz_compile.sh"
        self.model_name = model
        self.model = self.init_model(model)
        self.log = Log(filename="SRL")
        self.LDFLAGS, self.CFLAGS= read_bash_variables(f"{fuzz_dir}/compile.sh")
        self.compiler = find_compilers(f"{source_dir}/compile.sh")
        self.nop_feature = self.get_nop_feature()
        
        build_fuzz_directories(self.fuzz_dir)
        
        self.bb_file_path =  f"{fuzz_dir}/BasicBlock.txt"
        self.functions = parse_file(self.bb_file_path)
        
        
        # 示例输出,获取初始概率
        self.log.write(f"初始概率为:", "green")
        self.temp_functions = self.functions
        # 将temp输出到temp目录中
        output_file(self.temp_functions, self.temp_bb_file_path)
        self.init_probability_0, self.init_probability_1 = self.get_probability()
        self.adversarial_label = 0 if self.init_probability_0 < self.init_probability_1 else 1 # 哪个概率小，哪个就是对抗样本标签
        self.log.write(f"对抗样本label标签为:{self.adversarial_label}\n", "green")
        print(self.init_probability_0, self.init_probability_1)
    
    def run(self):
        graph_data = self.get_graph_data()
        out, before_classifier_output = self.model(graph_data)
        basic_block_count = graph_data.num_nodes
        if len(before_classifier_output.shape) != 2:
            print(f"before_classifier_output's shape is not 2 !")
            exit(1)
        input_dim = before_classifier_output.shape[1]
        output_dim = basic_block_count + 1 
        self.init_Qnetwork(input_dim,output_dim)
        
        action_nop_list = []
        attack_success = False
        t = 0
        state = graph_data 
        while self.get_label(state) != self.adversarial_label and t < self.iteration:
            if random.random() < self.epsilon:
                # 随机选择动作
                action_basicblock = select_random_blocks(basic_block_count)
                action_nop = random.choice(range(self.semantic_nops))
                
            else:
                # 使用Q网络选择动作
                out, before_classifier_output = self.model(state)
                q_values = self.q_network(torch.FloatTensor(before_classifier_output))
                
                # 获取基本块重要性
                bb_important_values = q_values[0, :-1]
                # 进行排序，并获取排序后的索引  # 从大到小排序
                sorted_indices = np.argsort(bb_important_values.detach().numpy())[::-1]
                # 获取排名在前topk的索引
                rank_topk_index = sorted_indices[:self.topk]  # 索引从0开始
                action_basicblock = rank_topk_index
                # 语义NOP指令的编号 让其取整数, 从0开始
                action_nop = int(q_values[0, -1].detach().numpy().item() * (self.semantic_nops-1))
                
            
            # 执行动作并得到新状态  
            action_nop_list.append(action_nop)              
            new_state = copy.deepcopy(state)  # 使用深拷贝
            for index in action_basicblock:
                new_state.x[index] += self.nop_feature.x[action_nop]
            # 计算奖励
            reward = 1 if self.get_probability_new(new_state)[self.adversarial_label] \
                        > self.get_probability_new(state)[self.adversarial_label] else 0
            # 将reward变成与q_values形状一致的张量
            reward = torch.tensor([reward], dtype=torch.float32).expand(self.output_dim)
            
            # 检查状态差异
            # if Diff(state.x, new_state.x) > self.delta:
            #     t = self.iteration
            #     reward = 0
            
            # 存储经验
            self.replay_buffer.add((state, action_nop, action_basicblock, reward, new_state))
            
            # 更新状态
            state = new_state
            t += 1
            
            # 定期更新Q网络参数
            if self.replay_buffer.size() > self.batch_size and t % self.T == 0:
                batch = self.replay_buffer.sample(self.batch_size)
                states, action_nop, action_basicblock, rewards, next_states =  zip(*batch)
                
                states = Batch.from_data_list(states)
                action_nop = torch.LongTensor(action_nop)
                action_basicblock = torch.LongTensor(np.array(action_basicblock))
                rewards = torch.FloatTensor(np.array(rewards))
                next_states = Batch.from_data_list(next_states)
                
                out, before_classifier_output = self.model(states)
                out, next_before_classifier_output = self.model(next_states)
                
                q_values = self.q_network(before_classifier_output)
                next_q_values = self.target_q_network(next_before_classifier_output)
                
                # 确保奖励和下一个Q值的形状一致
                rewards = rewards.view(next_q_values.shape)
                target_values = rewards + self.gamma * next_q_values
                
                loss = nn.MSELoss()(q_values, target_values)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 定期更新目标Q网络
            if t % self.C_update_freq == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        if self.get_label(state) != self.adversarial_label:
            attack_success = True # 攻击成功
            # 将动作序列输出到文件中
            seed_out_path = f"{fuzz_dir}/out/success_{self.model_name}_{t}.txt"
            with open(seed_out_path, 'w') as f:
                f.write(f"{action_nop_list}\n")
            
            self.log.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
            self.log.write(f"attack susccess \n\n")
            self.log.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
        
        else:
            seed_out_path = f"{fuzz_dir}/out/failed_{self.model_name}_{t}.txt"
            with open(seed_out_path, 'w') as f:
                f.write(f"{action_nop_list}\n")
            
        return attack_success
            
    def init_Qnetwork(self,input_dim, output_dim):
        self.input_dim = input_dim # before_classifier_output.shape[0] 是bachsize
        self.output_dim = output_dim # 输出维度，节点重要性 +语义NOP指令编号
        self.epsilon = 0.1
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.capacity = 1000 #缓冲区的容量
        self.batch_size = 3
        self.topk = 30
        self.iteration = 50
        self.delta = 0.1
        self.T = 3
        self.C_update_freq = 3
        self.semantic_nops = 27
        
        # 初始化Q网络和目标Q网络
        self.q_network = QNetwork(self.input_dim, self.output_dim)
        self.target_q_network = QNetwork(self.input_dim, self.output_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.capacity)
    
    def get_nop_feature(self):
        feature_size = self.model_name.split("_")[1]
        model_type = self.model_name.split("_")[0]
        if feature_size == '9':
            dataset = CFGDataset_Semantics_Preseving(root= "/home/lebron/IRFuzz/nop")
        elif feature_size == '20':
            dataset = CFGDataset_MAGIC_Attack(root= "/home/lebron/IRFuzz/nop")
        # 因为这里只有一个样本，所以一次循环就结束了
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=5)
        for data in data_loader:
            data = data.to(self.device)
        # 逐行打印
        # for row in data.x:
        #     print(row)
        return data
            
    def get_graph_data(self):
        feature_size = self.model_name.split("_")[1]
        model_type = self.model_name.split("_")[0]
        if feature_size == '9':
            dataset = CFGDataset_Semantics_Preseving(root= self.fuzz_dir)
        elif feature_size == '20':
            dataset = CFGDataset_MAGIC_Attack(root= self.fuzz_dir)
        else: 
            raise NotImplementedError
        
        # 因为这里只有一个样本，所以一次循环就结束了
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=5)
        for data in data_loader:
            data = data.to(self.device)
        return data
    
    def get_probability(self):
        # 插入+链接
        res = run_bash(script_path= self.bash_sh,
                args=[self.fuzz_dir, self.fuzz_dir, self.temp_bb_file_path, self.LDFLAGS, self.CFLAGS, self.compiler])
        if res == -1:
            print("run fuzz_compile.sh failed! Please check carefully!\n")
            exit()
            
        # 返汇编
        disassemble(fuzz_dir=self.fuzz_dir)
        # 提取cfg
        extract_cfg(fuzz_dir=self.fuzz_dir)
        # 模型预测
        data, result, prediction, before_classifier_output = measure(self.fuzz_dir, model=self.model_name) # prediction 0是良性 1是恶意  目前要把恶意转为良性。 result是模型输出的logsoftmax概率
        result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
        formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)
        probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
        probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
        self.log.write(f"probability_0 is {probability_0} probability_1:{probability_1} \n\n", "green")

        return probability_0,  probability_1
    
    def get_probability_new(self,data):

        data, result, prediction, before_classifier_output = self.get_output(data) # prediction 0是良性 1是恶意  目前要把恶意转为良性。 result是模型输出的logsoftmax概率
        result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
        formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)
        probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
        probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
        self.log.write(f"probability_0 is {probability_0} probability_1:{probability_1} \n\n", "green")
        return probability_0,  probability_1
        
    def init_model(self, model_name):
        feature_size = model_name.split("_")[1]
        model_type = model_name.split("_")[0]
        
        if model_type == "DGCNN":
            model = DGCNN(num_features=int(feature_size), num_classes=2)
        elif model_type == "GIN0":
            model = GIN0(num_features=int(feature_size), num_layers=4, hidden=64,num_classes=2)
        elif model_type == "GIN0WithJK":
            model = GIN0WithJK(num_features=int(feature_size), num_layers=4, hidden=64,num_classes=2)
        else:
            print(f"Model: {model_type} is not exist!")
            raise NotImplementedError
        
        model.load_state_dict(
            torch.load(f"/home/lebron/IRattack/py/model/record/{model_name}.pth", map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        return model
        
    def get_output(self, data):
        data = data.to(self.device)
        out, before_classifier_output = self.model(data)
        # print(out)
        result = torch.exp(out) # 将模型输出的logsoftmax转换为softmax
        formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)
        probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
        probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
        # print(f"probability_0: {probability_0}, probability_1: {probability_1}")
        # print(f"predictions: {out.argmax(dim=1).tolist()[0]}")

        return data, out, out.argmax(dim=1).tolist()[0], before_classifier_output
    
    def get_label(self, data):
        _,_,label,_ = self.get_output(data)
        return label

def is_success_file_present(fuzz_dir, model):
    success_files = glob.glob(f"{fuzz_dir}/out/success_{model}*")
    return len(success_files) > 0         

if __name__ == "__main__":
    
    model_list = ["DGCNN_9"]    
    
    ATTACK_SUCCESS_MAP = {
        "DGCNN_9":[],
        "DGCNN_20":[],
        "GIN0_9":[],
        "GIN0_20":[],
        "GIN0WithJK_9":[],
        "GIN0WithJK_20":[]
    }
    ATTACK_SUCCESS_RATE = dict()
    MAX_ITERATIONS = 30                       # 最大迭代次数
    LOGFILE = Log(filename="SRL_Time")   # 全局的日志文件
    
    malware_store_path = "/home/lebron/IRFuzz/ELF"
    malware_full_paths = [os.path.join(malware_store_path, entry) for entry in os.listdir(malware_store_path)]
    
    total_iterations = len(model_list) * len(malware_full_paths)
    progressed = 0

    for model in model_list:
        
        for malware_dir in malware_full_paths:
            source_dir= malware_dir
            fuzz_dir=  malware_dir
            model = model
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
            srl = SRL(fuzz_dir,model="DGCNN_9")
            attack_success = srl.run()                
            endtime =  datetime.datetime.now()
            
            if attack_success:
                LOGFILE.write(f"{model}-{source_dir.split('/')[-1]}\n")
                ATTACK_SUCCESS_MAP[model].append(source_dir.split('/')[-1])
                LOGFILE.write(f"Use {(endtime - startime).total_seconds()} s\n\n")
  
            progressed += 1
            
    for key in ATTACK_SUCCESS_MAP:
        ATTACK_SUCCESS_RATE[key] = len(ATTACK_SUCCESS_MAP[key]) / len(malware_full_paths)
    
    with open('/home/lebron/IRFuzz/attack_success_object.txt', 'w') as file:
        for key, object in ATTACK_SUCCESS_MAP.items():
            file.write(f'{key}: {str(object)}\n')  # 输出格式化的浮点数   
    
    with open('/home/lebron/IRFuzz/attack_success_rate.txt', 'w') as file:
        for key, value in ATTACK_SUCCESS_RATE.items():
            file.write(f'{key}: {value:.4f}\n')  # 输出格式化的浮点数
    exit()
    
    
    

