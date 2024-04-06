import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
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
        
        # 创建一个长度等于nop_len的向量，用0填充
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
    def __init__(self, basic_blocks_info, nop_list):
        self.basic_blocks_info = basic_blocks_info  # 基本块编号: {'inst_nums': 指令数量, 'nops_insert_count': {nop编号: 使用次数}}
        self.nop_list = nop_list                    #  {nop编号: 使用次数}
        self.visited_blocks = set()                 # 记录访问过的基本块
        self.used_nops = set()                      # 记录已使用的NOP
        self.blocks_keys_list = list(basic_blocks_info.keys())
        self.previous_probability_0 = 0

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

    def update_state(self, block_id, nop_id):
        self.visited_blocks.add(block_id)
        self.used_nops.add(nop_id)
        block_name = self.blocks_keys_list[block_id]
        self.basic_blocks_info[block_name]['nops_insert_count'][nop_id] += 1
        self.nop_list[nop_id] += 1

    def compute_reward(self, action):
        block_id, nop_id = action

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
        result, prediction = measure() # prediction 0是良性 1是恶意  目前要把恶意转为良性。 result是模型输出的logsoftmax概率
        result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
        probability_0 = result.tolist()[0][0] # 暂时先是一个样本
        if probability_0 > self.previous_probability_0: # 如果良性的概率增加，也可以增加reward
            self.previous_probability_0 = probability_0
            reward += 20
        
        if prediction == 0:
            reward += 100
            exit()
        else:
            reward += 0
        
        return reward

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
    def __init__(self, basic_blocks_info, nop_lists):
        super(ActorCritic, self).__init__()
        self.num_basic_blocks = len(basic_blocks_info)  # 示例基本块数量
        self.num_asm_instructions = len(nop_lists)  # 示例汇编指令数量
        self.basic_blocks_info = basic_blocks_info
        self.nop_lists = nop_lists
        
        self.actor = nn.Sequential(
            nn.Linear(self.num_basic_blocks * (self.num_asm_instructions + 1), 128), # 每个动作的离散表示
            nn.ReLU(),
            nn.Linear(128, self.num_basic_blocks * self.num_asm_instructions), # 输出每个动作的概率
            nn.Softmax(dim=-1)  # 确保输出是有效的概率分布---避免负数
        )
        self.critic = nn.Sequential(
            nn.Linear(self.num_basic_blocks * (self.num_asm_instructions + 1), 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 当前动作的分数
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

def train(basic_blocks_info, nop_lists):
    model = ActorCritic(basic_blocks_info, nop_lists)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    env = CFGEnvironment(basic_blocks_info, nop_lists)

    # 随机选择基本块索引和汇编指令索引
    # 这里把整个基本块的信息和插入信息作为了state
    state = dict_to_vector(basic_blocks_info)
    block_id = np.random.randint(0, model.num_basic_blocks)
    nop_id = np.random.randint(0, model.num_asm_instructions)
    # 在随机选中的位置置1
    model.basic_blocks_info[list(model.basic_blocks_info.keys())[block_id]]["nops_insert_count"][nop_id] +=  1
    state = dict_to_vector(model.basic_blocks_info)
    
    
    for episode in range(1000):
        print(f"Now is {episode}****************\n")
        # 二维数组 num_basic_blocks * num_asm_instructions ---整个action的可能空间
        # 把挑选出来的basic_block的序号 和 asminstructions的序号位置  置 1
        # 开始的时候是随机挑选的
        # state [num_basic_blocks ,num_asm_instruction + 1]  (之所以+1 是把基本块中的指令数 也作为特征了)
        action_probs, state_value = model(torch.FloatTensor(state).view(-1))
        # 扁平化概率分布，并从中选择一个动作
        flat_probs = action_probs.view(-1)  # 扁平化二维数组为一维
        num_rows, num_cols = model.num_basic_blocks, model.num_asm_instructions
        action = flat_probs.multinomial(1).detach()  # 根据概率选择动作
        # 将扁平化的索引映射回原始二维数组的行和列
        selected_index = action.item()
        block_id = selected_index // num_cols
        nop_id = selected_index % num_cols
        
        reward = env.compute_reward(action=(block_id, nop_id))  # 执行动作并获得奖励
        env.update_state(block_id,nop_id) # 更新状态
        model.basic_blocks_info = env.basic_blocks_info
        model.nop_lists = env.nop_list

        # 以下是简化的训练逻辑
        # 实际中需要处理动作概率、奖励、值函数等
        optimizer.zero_grad()
        loss = -torch.log(action_probs[action]) * (reward - state_value)
        loss.backward()
        optimizer.step()
        
        torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    
    
    
    pass