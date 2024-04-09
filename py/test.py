import copy
import os
import torch
from model.src.gnn import run_measure
from model.src.dataset_construct import run_disassemble
from model.src.dataset_construct import run_extract_cfg
from fuzz_attack import disassemble, extract_cfg, measure
from model.src.gnn.model import DGCNN
from torch_geometric.data import Data

def save_tensor_to_file(tensor, file_path):
    """
    将给定的Tensor按行保存到指定的文件路径。

    参数:
        tensor (torch.Tensor): 要保存的Tensor。
        file_path (str): 输出文件的路径。
    """
    # 检查并调整形状
    if tensor.shape[0] < tensor.shape[1]:
        # 维数较大的维度不在第0位，需要交换维度
        tensor = tensor.transpose(0, 1)

    # 保存到文件
    with open(file_path, 'w') as file:
        for row in tensor:
            line = ' '.join(map(str, row.tolist()))
            file.write(line + '\n')

def remove_random_edges(data, num_edges_to_remove):
    """
    随机移除图数据中指定数量的边，并返回被移除的边的信息。
    
    参数:
    - data: 包含edge_index成员的图数据对象。
    - num_edges_to_remove: 要随机移除的边的数量。
    
    返回:
    - 更新后的图数据对象。
    - 被移除的边的列表，每个元素是一个元组(start_node, end_node)。
    """
    num_edges = data.edge_index.shape[1]
    edges_to_remove = torch.randperm(num_edges)[:num_edges_to_remove].tolist()
    removed_edges = [(data.edge_index[0, idx].item(), data.edge_index[1, idx].item()) for idx in edges_to_remove]
    
    # 创建一个掩码，用于过滤掉被移除的边
    mask = torch.ones(num_edges, dtype=torch.bool)
    mask[edges_to_remove] = False
    new_edge_index = data.edge_index[:, mask]

    # 更新data的edge_index
    data.edge_index = new_edge_index

    return data, removed_edges

def remove_edges_sequentially(data):
    """
    生成器函数：按顺序逐条移除图数据中的边。

    参数:
    - data: 包含edge_index成员的图数据对象。

    产出:
    - 每次移除一条边后的图数据对象。
    """
    num_edges = data.edge_index.shape[1]
    for edge_to_remove in range(num_edges):
        # 打印出被移除的边
        removed_edge = data.edge_index[:, edge_to_remove]
        print(f"Removing edge: {removed_edge[0].item()} -> {removed_edge[1].item()}")

        # 移除选中的边
        new_edge_index = torch.cat((data.edge_index[:, :edge_to_remove], 
                                    data.edge_index[:, edge_to_remove+1:]), dim=1)

        # 创建一个新的Data对象，避免修改原始数据
        new_data = Data(x=data.x.clone(), edge_index=new_edge_index)
        
        # 如果data包含其他属性，也可以在这里复制
        
        yield new_data
        
def collect_probability_changes(data, model,file_path):
    """
    收集移除每条边之后，类别0概率的变化幅度。
    
    参数:
    - data: 包含edge_index成员的图数据对象。
    - model: 已经训练好的模型。
    
    返回:
    - 包含(边信息, 类别0概率变化幅度)的列表。
    """
    changes = []
    original_edges = [(data.edge_index[0][i].item(), data.edge_index[1][i].item()) 
                      for i in range(data.edge_index.shape[1])]
    original_data = copy.deepcopy(data)
    
    # 初始化模型并计算原始概率
    out_init = model(original_data)
    out_init = torch.exp(out_init)  # 将模型输出的log_softmax转换为softmax
    init_prob_0 = out_init[0][0].item()
    edge_remover = remove_edges_sequentially(data)

    for i, edge in enumerate(original_edges):
        # 移除边并计算概率
        data_modified = next(edge_remover)
        out = model(data_modified)
        out = torch.exp(out)  # 转换为softmax
        current_prob_0 = out[0][0].item()
        
        # 计算类别0概率的变化幅度并收集
        prob_change_0 = current_prob_0 - init_prob_0
        changes.append((f"{edge[0]} -> {edge[1]}", prob_change_0))

    save_changes_sorted(changes=changes,file_path=file_path, init_prob_0=init_prob_0)
    return changes

def save_changes_sorted(changes, file_path, init_prob_0):
    """
    将概率变化幅度从高到低排序并保存到文件。
    
    参数:
    - changes: 包含(边信息, 类别0概率变化幅度)的列表。
    - file_path: 输出文件的路径。
    """
    # 按类别0概率的变化幅度排序
    sorted_changes = sorted(changes, key=lambda x: x[1], reverse=True)
    
    with open(file_path, 'w') as file:
        file.write(f"Initial Class 0 Probability: {init_prob_0:.6f}\n\n")
        for edge, change in sorted_changes:
            file.write(f"Edge: {edge}, Class 0 Probability Change: {change:.6f}\n")

def collect_probability_changes_for_multiple_edges(data, model, num_edges_to_remove, episode,file_path):
    """
    移除随机数量的边，并收集类别0概率的变化幅度和移除的边的信息。
    """

    model.eval()
    with torch.no_grad():
        out_init = model(data)
        out_init = torch.exp(out_init)
        init_prob_0 = out_init[0][0].item()
    changes_with_edges=[]
    while episode:
        episode -= 1 
        copy_data = copy.deepcopy(data)
        data_modified, removed_edges = remove_random_edges(copy_data, num_edges_to_remove)
        with torch.no_grad():
            out_modified = model(data_modified)
            out_modified = torch.exp(out_modified)
            modified_prob_0 = out_modified[0][0].item()
            prob_change_0= modified_prob_0 - init_prob_0
            changes_with_edges.append((removed_edges, prob_change_0))

    save_changes_with_mutiedge_info(init_prob_0, changes_with_edges, file_path+f"_{num_edges_to_remove}")
    return removed_edges, init_prob_0

def save_changes_with_mutiedge_info(init_prob_0, changes_with_edges, file_path):
    """
    记录初始类别0的概率，并将概率变化和移除的边的信息保存到文件。
    """
    # 首先，按类别0概率的变化幅度排序
    changes_with_edges.sort(key=lambda x: x[1], reverse=True)
    
    with open(file_path, 'w') as file:
        file.write(f"Initial Class 0 Probability: {init_prob_0:.6f}\n\n")
        for removed_edges, prob_change in changes_with_edges:
            removed_edges_str = ", ".join(f"{start} -> {end}" for start, end in removed_edges)
            file.write(f"Removed Edges: {removed_edges_str}\n")
            file.write(f"Class 0 Probability Change: {prob_change:.6f}\n\n")

import torch
from torch_geometric.data import Data

def remove_edges_by_indices(data, indices_to_remove):
    """
    移除data.edge_index中指定位置处的边。
    
    参数:
    - data: 包含edge_index成员的图数据对象。
    - indices_to_remove: 一个整数列表，指定要移除的边的索引。
    
    返回:
    - 更新后的图数据对象。
    """
    # 创建一个掩码，初值为True
    mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
    
    # 将需要移除的边的位置设置为False
    mask[indices_to_remove] = False
    
    # 使用掩码过滤edge_index
    new_edge_index = data.edge_index[:, mask]

    # 创建一个新的Data对象，避免修改原始数据
    new_data = Data(x=data.x.clone(), edge_index=new_edge_index)
    
    # 如果data包含其他属性，也可以在这里复制
    # 例如: new_data.y = data.y.clone() 如果y是节点的标签

    return new_data


device = torch.device("cpu")
model = DGCNN(num_features=20, num_classes=2)
model.load_state_dict(
    torch.load("/home/lebron/IRattack/py/model/record/dgcnn_2.pth", map_location=device))
model = model.to(device)
model.eval()


record_file_path = "/home/lebron/IRattack/log/record_file.txt"
data = torch.load("/home/lebron/disassemble/attack/cfg_magic_test/data_{}_{}.pt".format(1, "server"))
predictions=[]

data = data.to(device)
modify_data = remove_edges_by_indices(data,[4,13])
result = model(modify_data)
predictions.extend(result.argmax(dim=1).tolist())
print(f"predictions: {predictions}")
result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)
probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
print(f"probability_0: {probability_0}, probability_1: {probability_1}")
exit()

# collect_probability_changes(data=data, model=model,file_path=record_file_path)
collect_probability_changes_for_multiple_edges(data=data,model=model,\
                                                file_path=record_file_path,episode=2000,num_edges_to_remove=20)
exit()



data_dir="/home/lebron/disassemble/attack/"
# disassemble(fuzz_dir=data_dir)
# extract_cfg(fuzz_dir=data_dir)
data, result, predictions=measure(fuzz_dir=data_dir)
result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)

probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本

exit()


# save_tensor_to_file(data.x, "/home/lebron/disassemble/attack/cfg_magic_test/data_{}_{}_x".format(0, "cub3"))
# save_tensor_to_file(data1.x, "/home/lebron/disassemble/attack/cfg_magic_test/data_{}_{}_x".format(1, "server"))
# save_tensor_to_file(data2.x, "/home/lebron/disassemble/attack/cfg_magic_test/data_{}_{}_x".format(2, "phide"))

# save_tensor_to_file(data.edge_index, "/home/lebron/disassemble/attack/cfg_magic_test/data_{}_{}_edge_index".format(0, "cub3"))
# save_tensor_to_file(data1.edge_index, "/home/lebron/disassemble/attack/cfg_magic_test/data_{}_{}_edge_index".format(1, "server"))
# save_tensor_to_file(data2.edge_index, "/home/lebron/disassemble/attack/cfg_magic_test/data_{}_{}_edge_index".format(2, "phide"))

# # data3 = torch.load("/home/lebron/IRFuzz/Linux.Phide/cfg_magic_test/data_{}_{}.pt".format(0, "phide"))
# # save_tensor_to_file(data3.edge_index, "/home/lebron/IRFuzz/Linux.Phide/cfg_magic_test/data_{}_{}_edge_index".format(0, "phide"))
# # save_tensor_to_file(data3.x, "/home/lebron/IRFuzz/Linux.Phide/cfg_magic_test/data_{}_{}_x".format(0, "phide"))

# data  = torch.load("/home/lebron/disassemble/attack/cfg_magic_test/data_{}_{}.pt".format(0, "cub3"))
# data1 = torch.load("/home/lebron/disassemble/attack/cfg_magic_test/data_{}_{}.pt".format(1, "server"))
