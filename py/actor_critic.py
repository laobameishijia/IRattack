import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def dict_to_vector(basic_blocks_info):
    # 找出指令数量和NOP使用次数的最大值
    max_instructions = max(block['instruction_count'] for block in basic_blocks_info.values())
    max_nops_count = max(max(block['nops_count'].values()) for block in basic_blocks_info.values() if block['nops_count'])

    vectors = []
    for block_id, block_data in basic_blocks_info.items():
        vector = [ block_data['instruction_count']/ max_instructions ]   # 开始构建向量并包含指令数量
        nop_len = len(block_data['nops_count'])  # 总共这么长的大小
        
        # 创建一个长度等于nop_len的向量，用0填充
        nops_vector = [0] * nop_len
        
        # 在正确的位置上设置nops_count的值
        for nop_id, count in block_data['nops_count'].items():
            nops_vector[nop_id] = count / max_nops_count
        
        # 将nops_vector合并到向量中
        vector.extend(nops_vector)
        
        # 将每个基本块的向量添加到结果列表中
        vectors.append(vector)
    
    return vectors


"""
basic_block_info = {
    'block1': {
        'instruction_count': 10,
        'nops_count': {
            0: 1,
            1: 2,
            2: 3
        }
    },
    'block2': {
        'instruction_count': 5,
        'nops_count': {
            0: 2,
            1: 1,
            2: 0
        }
    }
}

"""



"""
1. 优先选择没有插过的基本块
2. 优先选择没有插入过的asm
3. 优先选择asm指令数较少的
4. 优先选择基本块包含指令数较少的
5. 优先选择被选中的基本块中尚未插入过的asm指令
"""
class CFGEnvironment:
    def __init__(self, basic_blocks_info, nop_list):
        self.basic_blocks_info = basic_blocks_info  # 基本块编号: {'instruction_count': 指令数量, 'nops_count': {nop编号: 使用次数}}
        self.nop_list = nop_list                    #  {nop编号: 使用次数}
        self.visited_blocks = set()                 # 记录访问过的基本块
        self.used_nops = set()                      # 记录已使用的NOP

    def select_action(self):
        best_score = -float('inf')
        best_action = (None, None)  # (基本块编号, NOP编号)

        for block_id, block_info in self.basic_blocks_info.items():
            block_score = 0

            # 优先原则1: 检查基本块是否未被访问过
            if block_id not in self.visited_blocks:
                block_score += 10  # 未访问的基本块获得更高的分数

            # 优先原则4: 基于基本块包含的指令数
            block_score += 1 / (block_info['instruction_count'] + 1)

            for nop_id in range(len(self.nop_list)):
                action_score = block_score

                # 优先原则2: 检查NOP是否未被插入过
                if nop_id not in self.used_nops:
                    action_score += 5  # 未使用的NOP获得更高的分数

                # 优先原则3: 基于NOP的插入次数
                action_score += 1 / (block_info['nops_count'].get(nop_id, 0) + 1)

                # 优先原则5: 选中的基本块中未插入过的NOP
                if nop_id not in block_info['nops_count']:
                    action_score += 3

                if action_score > best_score:
                    best_score = action_score
                    best_action = (block_id, nop_id)

        return best_action

    def update_state(self, block_id, nop_id):
        self.visited_blocks.add(block_id)
        self.used_nops.add(nop_id)
        self.basic_blocks_info[block_id]['nops_count'][nop_id] = self.basic_blocks_info[block_id]['nops_count'].get(nop_id, 0) + 1

    def compute_reward(self, action):
        block_id, nop_id = action

        # 以最简单的形式计算奖励
        reward = 0
        if block_id not in self.visited_blocks:
            reward += 10
        if nop_id not in self.used_nops:
            reward += 5
        reward += 1 / (self.basic_blocks_info[block_id]['instruction_count'] + 1)
        reward += 1 / (self.basic_blocks_info[block_id]['nops_count'].get(nop_id, 0) + 1)

        
        # 这个里面要加上最终识别
        
        # 运行pass插入
        
        # 编译
        # 反汇编提取cfg
        # 模型预测
        
        if result_change:
            reward += 100
        else:
            reward += 0
            
        return reward



num_basic_blocks = len(basic_blocks_info)  # 示例基本块数量
num_asm_instructions = len(nop_lists)  # 示例汇编指令数量

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_basic_blocks * (num_asm_instructions + 1), 128), # 每个动作的离散表示
            nn.ReLU(),
            nn.Linear(128, num_basic_blocks * num_asm_instructions) # 输出每个动作的概率
        )
        self.critic = nn.Sequential(
            nn.Linear(num_basic_blocks * (num_asm_instructions + 1), 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 当前动作的分数
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

    def train():
        model = ActorCritic()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        env = CFGEnvironment(basic_blocks_info, nop_list)

        # 随机选择基本块索引和汇编指令索引
        # state = np.zeros((num_basic_blocks, num_asm_instructions))
        state = dict_to_vector(basic_blocks_info)
        block_id = np.random.randint(0, num_basic_blocks)
        asm_id = np.random.randint(0, num_asm_instructions)
        # 在随机选中的位置置1
        state[block_id, asm_id] = 1
        
        for episode in range(1000):
            # 二维数组 num_basic_blocks * num_asm_instructions ---整个action的可能空间
            # 把挑选出来的basic_block的序号 和 asminstructions的序号位置  置 1
            # 默认就随挑选
            action_probs, state_value = model(torch.FloatTensor(state))
            # 扁平化概率分布，并从中选择一个动作
            flat_probs = action_probs.view(-1)  # 扁平化二维数组为一维
            num_rows, num_cols = action_probs.shape
            action = flat_probs.multinomial(1).detach()  # 根据概率选择动作
            # 将扁平化的索引映射回原始二维数组的行和列
            selected_index = action.item()
            block_id = selected_index // num_cols
            asm_id = selected_index % num_cols
            
            reward = env.compute_reward(block_id, asm_id)  # 执行动作并获得奖励

            # 以下是简化的训练逻辑
            # 实际中需要处理动作概率、奖励、值函数等
            optimizer.zero_grad()
            loss = -torch.log(action_probs[action]) * (reward - state_value)
            loss.backward()
            optimizer.step()

            # 可以在这里添加逻辑来评估策略的性能，调整学习率等



exit()


for _ in range(iterations):
    action = env.select_action()  # 根据优先原则选择动作
    reward = env.compute_reward(action)  # 计算奖励
    env.update_state(*action)  # 更新环境状态

    # 在这里可以将(action, reward)用于更新你的强化学习模型


class QLearning:
    def __init__(self, num_states, num_actions, learning_rate=0.1, gamma=0.95):
        self.q_table = np.zeros((num_states, num_actions))
        self.learning_rate = learning_rate
        self.gamma = gamma

    def select_action(self, state):
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        next_max = np.max(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                      self.learning_rate * (reward + self.gamma * next_max)

# 假设num_states和num_actions根据环境设置
num_states = len(basic_blocks_info)  # 状态的数量可以是基本块的数量
num_actions = len(nop_list)  # 动作的数量可以是NOP指令的数量

# 初始化Q-learning模型
q_learning_model = QLearning(num_states, num_actions)

# 迭代学习过程
for _ in range(1000):  # 例如，迭代1000次
    state = np.random.randint(num_states)  # 假设每次迭代随机选择一个状态进行学习
    action = q_learning_model.select_action(state)
    
    # 假设有方法将状态和动作转换为环境可以接受的格式
    block_id, nop_id = translate_state_action_to_env(state, action)
    
    # 执行动作并获取奖励
    reward = env.compute_reward((block_id, nop_id))
    
    # 假设有方法获取下一个状态
    next_state = get_next_state(env, block_id, nop_id)
    
    # 更新Q表
    q_learning_model.update(state, action, reward, next_state)




exit()

class Env:
    
    def __init__(self, blocks_info, asms_info):
        self.blocks_info = blocks_info  # set()类型  每个基本块中的指令数量
        self.asms_info = asms_info      # set()类型  每个汇编指令被使用的次数
        self.visited_blocks = set()     # 已经插入的基本块名称集合
        self.used_asms = set()          # 使用过的汇编指令集合


    def step(self, action):
        block_id, asm_id = action

        # 优先级1: 优先选择没有探索过的基本块
        if block_id not in self.visited_blocks:
            reward = 10
            self.visited_blocks.add(block_id)
        else:
            reward = 1

        # 优先级2: 优先选择没有插入过的asm
        if asm_id not in self.used_asms:
            reward += 5
            self.used_asms.add(asm_id)

        # # 优先级3: 优先选择asm指令数较少的
        # # 假设asm_info存储了每个汇编指令使用次数
        # asm_usage = self.asms_info[asm_id]
        # reward += (1 / (asm_usage + 1)) * 2  # 为了简化，这里使用了倒数来计算奖励

        # # 优先级4: 优先选择基本块包含指令数较少的
        # block_size = self.blocks_info[block_id]
        # reward += (1 / (block_size + 1)) * 2  # 同样使用倒数来计算奖励
        
        # 优先级3: 基于使用次数的排名
        asm_usage_rank = sorted(self.asms_info, key=self.asms_info.get).index(asm_id)
        reward += 1 / (asm_usage_rank + 1)

        # 优先级4: 基于基本块大小的排名
        block_size_rank = sorted(self.blocks_info, key=self.blocks_info.get).index(block_id)
        reward += 1 / (block_size_rank + 1)


        return reward
    




"""这种方式每个恶意代码文件都需要训练一个模型，这样工作量太大了"""

# # 假设基本块和汇编指令编号已经被映射到连续的整数空间
# num_basic_blocks = 10  # 示例基本块数量
# num_asm_instructions = 20  # 示例汇编指令数量

# class ActorCritic(nn.Module):
#     def __init__(self):
#         super(ActorCritic, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Linear(num_basic_blocks + num_asm_instructions, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_basic_blocks * num_asm_instructions)
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(num_basic_blocks + num_asm_instructions, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, state):
#         action_probs = self.actor(state)
#         state_value = self.critic(state)
#         return action_probs, state_value

#     def train():
#         model = ActorCritic()
#         optimizer = optim.Adam(model.parameters(), lr=0.01)
#         env = Env()

#         for episode in range(1000):
#             state = ...  # 获取当前状态的表示
#             action_probs, state_value = model(torch.FloatTensor(state))
#             action = action_probs.multinomial(1).detach()  # 根据概率选择动作
#             reward = env.step(action.item())  # 执行动作并获得奖励

#             # 以下是简化的训练逻辑
#             # 实际中需要处理动作概率、奖励、值函数等
#             optimizer.zero_grad()
#             loss = -torch.log(action_probs[action]) * (reward - state_value)
#             loss.backward()
#             optimizer.step()

#             # 可以在这里添加逻辑来评估策略的性能，调整学习率等
