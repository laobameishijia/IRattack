import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import copy

from tqdm import tqdm
from model.src.dataset_construct import run_disassemble
from model.src.dataset_construct import run_extract_cfg
from model.src.gnn import run_measure

def dict_to_vector(basic_blocks_info):
    # 找出指令数量和NOP使用次数的最大值
    max_instructions = max(block['inst_nums'] for block in basic_blocks_info.values())
    max_nops_count = max(max(block['nops_insert_count'].values()) for block in basic_blocks_info.values() if block['nops_insert_count'])

    vectors = []
    for block_id, block_data in basic_blocks_info.items():
        vector = [ block_data['inst_nums']/ max_instructions ]   # 开始构建向量并包含指令数量
        nop_len = len(block_data['nops_insert_count'])  # 总共这么长的大小
        
        # 创建一个长度等于nop_len的向量,用0填充
        nops_vector = [0] * nop_len
        
        # 在正确的位置上设置nops_count的值
        for nop_id, count in block_data['nops_insert_count'].items():
            if max_nops_count:
                nops_vector[nop_id] = count / max_nops_count
            else:
                 nops_vector[nop_id] = 0
        
        # 将nops_vector合并到向量中
        vector.extend(nops_vector)
        
        # 将每个基本块的向量添加到结果列表中
        vectors.append(vector)
    
    return vectors

def run_bash(script_path, args:list):
    import subprocess
    print(f"Now run {script_path} \n")
    # 运行脚本
    result = subprocess.run([script_path] + args, text=True)
    # 等待子进程结束  打印脚本的输出
    # result.wait()
    print(f"exit {script_path} \n")

def read_result_file():
    filename = "/home/lebron/disassemble/attack/result.txt"
    # 读取文件
    with open(filename, 'r') as file:
        content = file.read()
    # result文件中的内容是 "predictions: 1"
    # 使用字符串分割来获取数字部分
    parts = content.split(':')
    if len(parts) > 1:
        # 尝试将分割后的第二部分转换为整数
        try:
            number = int(parts[1].strip())  # strip() 去除可能的空白字符
            print(f"The number is {number}")
            return number
        except ValueError as e:
            print(f"Error converting to integer: {e}")

def disassemble():
    dataset_dir="/home/lebron/disassemble/attack"
    dir_path = f"{dataset_dir}/rawdata"
    output_dir = f"{dataset_dir}/asm"
    log_path = f"{dataset_dir}/disassemble.log"
    run_disassemble.run(dir_path=dir_path,output_dir=output_dir,log_path=log_path)

def extract_cfg():
    dataset_dir= "/home/lebron/disassemble/attack"
    data_dir = f"{dataset_dir}/asm"
    store_dir = f"{dataset_dir}/cfg"
    file_format = "json"
    log_file = f"{dataset_dir}/extract_cfg.log"
    run_extract_cfg.run(data_dir=data_dir,store_dir=store_dir,file_format=file_format,log_file=log_file)

def measure():
    data_dir ="/home/lebron/disassemble/attack"
    result = run_measure.measure(data_dir=data_dir)
    return result
    
"""
1. 优先选择没有插过的基本块
2. 优先选择没有插入过的asm
3. 优先选择asm指令数较少的
4. 优先选择基本块包含指令数较少的
5. 优先选择被选中的基本块中尚未插入过的asm指令
"""
class CFGEnvironment:
    def __init__(self, basic_blocks_info, nop_list, log_file_path):
        self.basic_blocks_info = basic_blocks_info  # 基本块编号: {'inst_nums': 指令数量, 'nops_insert_count': {nop编号: 使用次数}}
        self.nop_list = nop_list                    #  {nop编号: 使用次数}
        self.visited_blocks = set()                 # 记录访问过的基本块
        self.used_nops = set()                      # 记录已使用的NOP
        self.blocks_keys_list = list(basic_blocks_info.keys())
        self.previous_probability_0 = 0
        self.previous_probability_1 = 1
        self.counting_rewards = 0 # 表明计算了多少次reward了
        self.log_file = open(log_file_path, "w")
        self.log_file.write("probability_0      probability_1\n")
        
        self.original_basic_blocks_info = copy.deepcopy(basic_blocks_info)
        self.original_nop_list = copy.deepcopy(nop_list)
        
    def reset(self):
        self.basic_blocks_info = self.original_basic_blocks_info
        self.nop_list = self.original_nop_list
        self.visited_blocks = set()                 # 记录访问过的基本块
        self.used_nops = set()                      # 记录已使用的NOP
        self.blocks_keys_list = list(self.basic_blocks_info.keys())
        self.previous_probability_0 = 0
        self.previous_probability_1 = 1
        self.counting_rewards = 0 # 表明计算了多少次reward了
        
        
    def select_action(self):
        best_score = -float('inf')
        best_action = (None, None)  # (基本块编号, NOP编号)

        for block_id, block_info in self.basic_blocks_info.items():
            block_score = 0

            # 优先原则1: 检查基本块是否未被访问过
            if block_id not in self.visited_blocks:
                block_score += 10  # 未访问的基本块获得更高的分数

            # 优先原则4: 基于基本块包含的指令数
            block_score += 1 / (block_info['inst_nums'] + 1)

            for nop_id in range(len(self.nop_list)):
                action_score = block_score

                # 优先原则2: 检查NOP是否未被插入过
                if nop_id not in self.used_nops:
                    action_score += 5  # 未使用的NOP获得更高的分数

                # 优先原则3: 基于NOP的插入次数
                action_score += 1 / (block_info['nops_insert_count'].get(nop_id, 0) + 1)

                # 优先原则5: 选中的基本块中未插入过的NOP
                if nop_id not in block_info['nops_insert_count']:
                    action_score += 3

                if action_score > best_score:
                    best_score = action_score
                    best_action = (block_id, nop_id)

        return best_action

    def update_state(self, action):
        block_id, nop_id = action
        self.visited_blocks.add(block_id)
        self.used_nops.add(nop_id)
        block_name = self.blocks_keys_list[block_id]
        self.basic_blocks_info[block_name]['nops_insert_count'][nop_id] += 1
        self.nop_list[nop_id] += 1

    def compute_reward(self, action):
        block_id, nop_id = action
        done = False
        
        # 以最简单的形式计算奖励
        reward = 0
        if block_id not in self.visited_blocks:
            reward += 10
        if nop_id not in self.used_nops:
            reward += 5
        reward += 1 / (self.basic_blocks_info[self.blocks_keys_list[block_id]]['inst_nums'])
        reward += 1 / (self.basic_blocks_info[self.blocks_keys_list[block_id]]['nops_insert_count'].get(nop_id, 0) + 1)

        
        # 这个里面要加上最终识别
        
        # 运行pass插入 + 编译
        run_bash(script_path="/home/lebron/disassemble/attack/sourcecode/Linux.Apachebd/attack/insertasminstruction.sh",
                 args=[self.blocks_keys_list[block_id], str(nop_id)])
        
        # 反汇编提取cfg + 模型预测
        # run_bash(script_path="/home/lebron/MCFG_GNN/src/dataset_construct/attack.sh",
        #     args=[])
        disassemble()
        extract_cfg()
        next_state, result, prediction = measure() # prediction 0是良性 1是恶意  目前要把恶意转为良性。 result是模型输出的logsoftmax概率
        result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
        formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)

        probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
        probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
        self.log_file.write(f"{probability_0}  {probability_1}\n") #将概率输出到文件中
        
        # 如果良性的概率增加,也可以增加reward
        if self.counting_rewards and probability_0 > self.previous_probability_0: 
            self.previous_probability_0 = probability_0
            reward += 20
        # 如果良性的概率增加 且 恶意的概率减少,也可以增加reward。增长幅度更大
        if self.counting_rewards and probability_0 > self.previous_probability_0 and probability_1 < self.previous_probability_1: 
            self.previous_probability_0 = probability_0
            reward += 50
        
        if prediction == 0:
            reward += 100
            exit()
        else:
            reward += 0
        
        self.counting_rewards += 1
        if self.counting_rewards == 100:
            done = True
        
        return next_state,reward, done

"""
basic_block_info = {
    'block1': {
        'inst_nums': 10,
        'nops_insert_count': {
            0: 1,
            1: 2,
            2: 3
        }
    },
    'block2': {
        'inst_nums': 5,
        'nops_insert_count': {
            0: 2,
            1: 1,
            2: 0
        }
    }
}

"""
class ActorCritic(nn.Module):
    def __init__(self, basic_blocks_info, nop_lists, gamma, actor_lr, critic_lr, device):
        super(ActorCritic, self).__init__()
        self.num_basic_blocks = len(basic_blocks_info)  # 示例基本块数量
        self.num_asm_instructions = len(nop_lists)  # 示例汇编指令数量
        self.basic_blocks_info = basic_blocks_info
        self.nop_lists = nop_lists
        self.device = device
        self.gamma = gamma
        
        # TODO: 这个地方暂时先写成死的, 因为反汇编统计出来的基本块数量和IR统计出来的基本块数量不一样！
        state_dim = 20 * 36 # 基本块特征是20维的, 暂时先不考虑  边矩阵
        action_dim = len(basic_blocks_info) + 26 # 要插入的基本块序数 + 汇编指令序数
        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), # 每个动作的离散表示
            nn.ReLU(),
            nn.Linear(128, action_dim), # 输出每个动作的概率
            nn.Softmax(dim=-1)  # 确保输出是有效的概率分布---避免负数
        )
        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 当前动作的分数
        )
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        # 价值网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def forward(self, state):
        action_probs = self.actor(state.view(-1))
        state_value = self.critic(state.view(-1))
        return action_probs, state_value
    
    def take_action(self, state):
        if state == None:
            block_id = np.random.randint(0, self.num_basic_blocks)
            nop_id = np.random.randint(0, self.num_asm_instructions)
            return block_id, nop_id
        
        state = state.x # 现在不考虑边之间的关系，因为我们也没有破坏边，边之间的关系是不变的
        action_probs, state_value = self.forward(torch.tensor(state))
        # 分割 action_probs 为基本块序数概率和汇编指令序数概率
        basic_blocks_probs = action_probs[:self.num_basic_blocks]
        asm_instructions_probs = action_probs[self.num_basic_blocks:]
        # 根据概率分布选择基本块序数和汇编指令序数
        # selected_basic_block = np.random.choice(len(basic_blocks_probs), p=basic_blocks_probs.detach().numpy() )
        # selected_asm_instruction = np.random.choice(26, p=asm_instructions_probs.detach().numpy() )
        # 假设 basic_blocks_probs 和 asm_instructions_probs 是 PyTorch 张量
        selected_basic_block = torch.multinomial(basic_blocks_probs, 1).item()
        selected_asm_instruction = torch.multinomial(asm_instructions_probs, 1).item()
        block_id = selected_basic_block
        nop_id = selected_asm_instruction
        return block_id, nop_id
   
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        
        """
        在强化学习中,通常情况下,当智能体与环境进行交互时,会观察到一个连续的状态序列。
        当智能体执行某个动作后,环境会返回下一个状态、奖励以及一个布尔值表示当前状态是否为episode的最后一个状态。
        如果当前状态为episode的最后一个状态,则这个布尔值通常被设置为True,表示当前episode结束了。
        """
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(
            torch.nn.functional.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
        
    def train(self, num_episodes):
        env = CFGEnvironment(self.basic_blocks_info, self.nop_lists, "/home/lebron/IRattack/probability_log")
        return_list = []
        for i in range(10):
            with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes/10)):
                    episode_return = 0
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                    state = env.reset()
                    done = False
                    # 在强化学习中，通常情况下，当智能体与环境进行交互时，会观察到一个连续的状态序列。
                    # 当智能体执行某个动作后，环境会返回下一个状态、奖励以及一个布尔值表示当前状态是否为episode的最后一个状态。
                    # 如果当前状态为episode的最后一个状态，则这个布尔值通常被设置为True，表示当前episode结束了。
                    while not done:
                        action = self.take_action(state) # action: (block_id, nop_id)
                        env.update_state(action)
                        next_state, reward, done = env.compute_reward(action)
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        state = next_state
                        episode_return += reward
                    return_list.append(episode_return)
                    self.update(transition_dict) # 更新参数
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
        return return_list
    

def train(basic_blocks_info, nop_lists):
    self = ActorCritic(basic_blocks_info, nop_lists)
    optimizer = optim.Adam(self.parameters(), lr=0.01)
    env = CFGEnvironment(basic_blocks_info, nop_lists, "/home/lebron/IRattack/probability_log")

    # 随机选择基本块索引和汇编指令索引
    # 这里把整个基本块的信息和插入信息作为了state
    state = dict_to_vector(basic_blocks_info)
    block_id = np.random.randint(0, self.num_basic_blocks)
    nop_id = np.random.randint(0, self.num_asm_instructions)
    # 在随机选中的位置置1
    self.basic_blocks_info[list(self.basic_blocks_info.keys())[block_id]]["nops_insert_count"][nop_id] +=  1
    state = dict_to_vector(self.basic_blocks_info)
    
    
    for episode in range(1000):
        print(f"Now is {episode}****************\n")
        # 二维数组 num_basic_blocks * num_asm_instructions ---整个action的可能空间
        # 把挑选出来的basic_block的序号 和 asminstructions的序号位置  置 1
        # 开始的时候是随机挑选的
        # state [num_basic_blocks ,num_asm_instruction + 1]  (之所以+1 是把基本块中的指令数 也作为特征了)
        action_probs, state_value = self(torch.FloatTensor(state).view(-1))
        # 扁平化概率分布,并从中选择一个动作
        flat_probs = action_probs.view(-1)  # 扁平化二维数组为一维
        num_rows, num_cols = self.num_basic_blocks, self.num_asm_instructions
        action = flat_probs.multinomial(1).detach()  # 根据概率选择动作
        # 将扁平化的索引映射回原始二维数组的行和列
        selected_index = action.item()
        block_id = selected_index // num_cols
        nop_id = selected_index % num_cols
        
        reward = env.compute_reward(action=(block_id, nop_id))  # 执行动作并获得奖励.
        env.update_state(block_id,nop_id) # 更新状态
        self.basic_blocks_info = env.basic_blocks_info
        self.nop_lists = env.nop_list

        # 以下是简化的训练逻辑
        # 实际中需要处理动作概率、奖励、值函数等
        optimizer.zero_grad()
        loss = -torch.log(action_probs[action]) * (reward - state_value)
        loss.backward()
        optimizer.step()
        
        torch.save(self.state_dict(), 'self.pth')
    
    # 指定要写入的文件路径
    file_path = "/home/lebron/IRattack/basic_blocks_info.json"

    # 将字典写入JSON文件
    with open(file_path, "w") as json_file:
        json.dump(self.basic_blocks_info, json_file)

if __name__ == "__main__":
    
    
    
    pass