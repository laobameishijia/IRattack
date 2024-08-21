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
from torch.nn import Conv1d, MaxPool1d
import torch.optim as optim
from torch_geometric.data import Data, Batch
import numpy as np
from collections import deque

import tqdm

from model.src.gnn.model import GIN0WithJK, GIN0, DGCNN
from model.src.gnn.dataset import CFGDataset_Semantics_Preseving, CFGDataset_MAGIC,CFGDataset_MAGIC_Attack
from torch_geometric.loader import DataLoader

from torch_geometric.nn import GCNConv
from model.src.gnn.sortaggregation.CustomSortAggregation import SortAggregation

# 定义Q网络---使用DGCNN来进行
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=27, k=32):
        super(QNetwork, self).__init__()
        self.k = k 
        
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        self.conv4 = GCNConv(32, 32)
        
        self.conv5 = Conv1d(1, 16, 256, 256)
        self.conv6 = Conv1d(16, 32, 5, 1)
        self.pool = MaxPool1d(2, 2)
        self.dense = nn.Linear(384, output_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()  # 添加sigmoid激活函数

    def forward(self, data):
        x, edge_index, batch =  data.x, data.edge_index, data.batch
        
        x_1 = torch.tanh(self.conv1(x, edge_index))
        x_2 = torch.tanh(self.conv2(x_1, edge_index))
        x_3 = torch.tanh(self.conv3(x_2, edge_index))
        x_4 = torch.tanh(self.conv4(x_3, edge_index))
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1) # (99, 256)
        x, top_k_indices = SortAggregation(k=self.k)(x, batch) # (1,8192)
        x = x.view(x.size(0), 1, x.size(-1)) # (1,1,16384)
        x = self.relu(self.conv5(x)) # (1,16,32)
        x = self.pool(x) # (1,16,16)
        x = self.relu(self.conv6(x)) #(1,32,12)
        x = x.view(x.size(0), -1) #（1,384)
        x = self.dense(x)
        # x = self.sigmoid(x)  # 应用sigmoid激活函数
        return x, top_k_indices # x为(0,1)之间的值，用于计算语义nop指令编号。top_k_indices是排序在前topK的节点序号，如果节点数目不足32,用-1填充了。

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
    
    def __init__(self, fuzz_dir="/home/lebron/IRFuzz/SRL" , filename="SRL"):
        self.log_file = open(f"{fuzz_dir}/{filename}", mode="a")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"Log file created at: {current_time}\n")
    
    def write(self, message, color="white"):
        self.log_file.write(f"{message}\n")
        print(colored(message, color))
        self.log_file.flush()

# 计算状态差异
def Diff(s1, s2):
    return np.linalg.norm(s1 - s2)

def select_random_blocks(basic_block_count, num_select=30):
    blocks = list(range(basic_block_count))
    if basic_block_count <= num_select:
        return blocks
    else:
        return random.sample(blocks, num_select)

def is_success_file_present(fuzz_dir, model):
    success_files = glob.glob(f"{fuzz_dir}/out/success_{model}*")
    return len(success_files) > 0         

class SRLAttack():
    
    def __init__(self, model, data_dir):

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model_name = model
        self.model = self.init_model(model)
        
        self.result_log = Log(filename=f"SRL_result_{self.model_name}")
        self.best_result = 0
        
        self.nop_feature = self.get_nop_feature()
        feature_size = self.model_name.split("_")[1]
        self.init_Qnetwork(input_dim=int(feature_size))
        self.init_dataset(data_dir)
        self.data_dir = data_dir

    def run(self):
        
        while self.train_iteration > 0:
            self.train_iteration -= 1
            
            data_index = 0              # 特征数据的编号
            self.success_data_num = []  # 攻击成功的样本数量-列表
            # self.load_best_model()      # 加载之前最好的模型，从最好的出发开始训练
            
            for data in tqdm.tqdm(self.data_loader):
                
                _, _, label,_, = self.get_output(data)
                self.adversarial_label = 0 if label == 1 else 1
                
                basic_block_count = data.num_nodes
            
                action_nop_list = [] # 保留选择的变异策略
                t = 0
                state = data 
            
                while self.get_label(state) != self.adversarial_label and t < self.iteration:
                    
                    epsilon = max(self.epsilon_end, \
                        self.epsilon_start - (self.step / self.epsilon_decay_steps) * (self.epsilon_start - self.epsilon_end))

                    if random.random() < epsilon:
                        # 随机选择动作
                        action_basicblock = select_random_blocks(basic_block_count)
                        action_nop = random.choice(range(self.semantic_nops))
                        print(f"random select action_nop is {action_nop}")
                    else:
                        # 使用Q网络选择动作
                        q_values,top_k_indices = self.q_network(state)
                        # 获取排名在前topk的索引
                        action_basicblock = top_k_indices.tolist()[0]
                        # 语义NOP指令的编号 取q_values最大值的编号
                        action_nop = torch.argmax(q_values).detach().item()
                        print(f"Q-network select action_nop is {action_nop}")
                        
                    # 执行动作并得到新状态  
                    action_nop_list.append(action_nop)              
                    new_state = copy.deepcopy(state)  # 使用深拷贝
                    for index in action_basicblock:
                        new_state.x[index] += self.nop_feature.x[action_nop]
                    # 计算奖励  将reward扩展为27维
                    reward = 1 if  self.probality_adversarial_rise(new_state,state) else 0
                    print(f"reward is {reward}")
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
                    self.step += 1
                    
                    # 定期更新Q网络参数
                    if  t % self.T == 0:
                        batch = self.replay_buffer.sample(self.batch_size)
                        states, action_nop, action_basicblock, rewards, next_states =  zip(*batch)
                        
                        states = Batch.from_data_list(states)
                        # action_nop = torch.LongTensor(action_nop)
                        # action_basicblock = torch.LongTensor(action_basicblock)
                        rewards = torch.cat(rewards)
                        rewards = rewards.view(len(next_states),self.output_dim)
                        next_states = Batch.from_data_list(next_states)
                        
                        
                        q_values, _ = self.q_network(states)
                        next_q_values, _ = self.target_q_network(next_states)
                        
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
                
                if self.get_label(state) == self.adversarial_label:
                    self.mutation_log.write(f"{data_index}-success!")
                    self.mutation_log.write(f"{action_nop_list}")
                    
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    print("attack susccess")
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    
                    self.success_data_num.append(data_index)
                    
                data_index += 1
            
            attack_success_rate = (len(self.success_data_num)/ len(self.data_loader))*100
            if attack_success_rate > self.best_result:
                self.best_result = attack_success_rate
                self.result_log.write(f"{attack_success_rate:.2f}")
                self.save_model()
            self.mutation_log.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            self.mutation_log.write(f"attack success rate {attack_success_rate:.2f}%\n")  
            self.mutation_log.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")  
        
        return 1
    
    def evaluate(self):
    
        self.load_best_model()      # 加载之前最好的模型
        self.q_network.eval()
        self.target_q_network.eval()
        iteration_list = [10, 20, 30, 40, 50, 60]
        # iteration_list = [10]
        
        for iteration in iteration_list:
            data_index = 0              # 特征数据的编号
            self.success_data_num = []  # 攻击成功的样本数量-列表
            self.mutation_log = Log(filename=f"SRL_mutation_{self.model_name}_{iteration}")
            for data in tqdm.tqdm(self.data_loader):
                
                _, _, label,_, = self.get_output(data)
                self.adversarial_label = 0 if label == 1 else 1
                
                action_nop_list = [] # 保留选择的变异策略
                t = 0
                state = data 
                startime =  datetime.datetime.now()
                while self.get_label(state) != self.adversarial_label and t < iteration:

                    # 使用Q网络选择动作
                    q_values,top_k_indices = self.q_network(state)
                    # 获取排名在前topk的索引
                    action_basicblock = top_k_indices.tolist()[0]
                    # 语义NOP指令的编号 取q_values最大值的编号
                    action_nop = torch.argmax(q_values).detach().item()
                    print(f"Q-network select action_nop is {action_nop}")
                    
                    # 执行动作并得到新状态  
                    action_nop_list.append(action_nop)              
                    new_state = copy.deepcopy(state)  # 使用深拷贝
                    for index in action_basicblock:
                        new_state.x[index] += self.nop_feature.x[action_nop]

                    
                    # # 检查状态差异
                    # if Diff(state.x, new_state.x) > self.delta:
                    #     break
                    # 更新状态
                    state = new_state
                    t += 1
                    self.step += 1
                
                if self.get_label(state) == self.adversarial_label:
                    endtime =  datetime.datetime.now()
                    self.mutation_log.write(f"{data_index}-success!")
                    self.mutation_log.write(f"{action_nop_list}")
                    self.mutation_log.write(f"Use {(endtime - startime).total_seconds()} s\n\n")
                    print("attack susccess")
                    self.success_data_num.append(data_index)
                    
                data_index += 1
        
            attack_success_rate = (len(self.success_data_num)/ len(self.data_loader))*100
            self.result_log.write(f"{iteration}----attack success rate {attack_success_rate:.2f}%\n")  
        
        return 1

    
    def save_model(self):
        torch.save(self.q_network.state_dict(), f"{self.data_dir}/train/model_{self.model_name}")
 
    def init_dataset(self, data_dir):
        feature_size = self.model_name.split("_")[1]
        if feature_size == '9':
            self.dataset = CFGDataset_Semantics_Preseving(root= data_dir)
        elif feature_size == '20':
            self.dataset = CFGDataset_MAGIC_Attack(root= data_dir)
        else: 
            raise NotImplementedError
        
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=5)
        
    def init_Qnetwork(self,input_dim):
        self.train_iteration = 100    # 要在整个样本集上训练多少轮
        self.input_dim = input_dim  # before_classifier_output.shape[0] 是bachsize
        self.output_dim = 27        # 输出维度语义NOP指令编号，节点重要性不包含在输出里面
        self.gamma = 0.99           # 未来期望的比重
        self.learning_rate = 0.001  # 学习率
        self.capacity = 3000        # 缓冲区的容量
        self.batch_size = 5         # 取多少个缓冲区的样本来更新数据
        self.topk = 30              # topk节点的数量
        self.iteration = 30         # 迭代次数
        self.delta = 0.05            # 暂时无用
        self.T = 10                 # 更新Q网络的频率
        self.C_update_freq = 30     # 更新目标网络的频率
        self.semantic_nops = 27     # 语义NOP指令的数量
        
        
        self.epsilon_start = 1      # 随机选择action的概率，从1-0.1进行衰减
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 3000
        self.step = 0
        
        # 初始化Q网络和目标Q网络
        self.q_network = QNetwork(self.input_dim)
        self.target_q_network = QNetwork(self.input_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.capacity)
    
    def load_best_model(self):
        file_path = f"{self.data_dir}/train/model_{self.model_name}"
        if os.path.exists(file_path):
            self.q_network.load_state_dict(torch.load(file_path))
            self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def probality_adversarial_rise(self,new_state,state):
        new = self.get_probability_new(new_state)[self.adversarial_label]
        old = self.get_probability_new(state)[self.adversarial_label]
        print(f"probality_adversarial: new_state-->{new}, state-->{old}")
        return  new > old  
    
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
            
    def get_probability_new(self,data):

        data, result, prediction, before_classifier_output = self.get_output(data) # prediction 0是良性 1是恶意  目前要把恶意转为良性。 result是模型输出的logsoftmax概率
        result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
        formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)
        probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
        probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
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
        # out.argmax(dim=1).tolist()[0] 是 prediction
        return data, out, out.argmax(dim=1).tolist()[0], before_classifier_output
    
    def get_label(self, data):
        _,_,label,_ = self.get_output(data)
        return label


if __name__ == "__main__":
    
    model_list = ["DGCNN_9","DGCNN_20","GIN0_9","GIN0_20","GIN0WithJK_9","GIN0WithJK_20"]    
    data_dir = "/home/lebron/IRFuzz/SRL"
    
    for model in model_list:
        srl = SRLAttack(model=model, data_dir=data_dir)
        srl.evaluate()

    
    
    

