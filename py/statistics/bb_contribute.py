import copy
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
import tqdm
from model.src.gnn import run_measure
from model.src.dataset_construct import run_disassemble
from model.src.dataset_construct import run_extract_cfg
from fuzz_attack import disassemble, extract_cfg, measure
from model.src.gnn.model import DGCNN
from model.src.gnn.dataset import CFGDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def format_asm(cfg, block_k_index):

    addr_to_id = dict() # {str: int}
    id_to_addr = dict() # {int: str} 将 基本块序号 和 基本块地址 联系起来
    current_node_id = -1

    for addr, block in cfg.items():
        current_node_id += 1
        addr_to_id[addr] = current_node_id
        id_to_addr[current_node_id] = addr
    
    asm_str = ""
    for index in block_k_index:
        block = cfg[id_to_addr[index]]
        asm_str += f"{index}---{id_to_addr[index]}; Block starts at address {hex(block['start_addr'])} and ends at address {hex(block['end_addr'])}\n"
        for insn in block['insn_list']:
            addr = hex(insn['address'])
            operands = ', '.join(insn['operands'])
            asm_str += f"{addr}: {insn['opcode']} {operands} \n"
        asm_str += "\n"
        
    asm_str += "\n; Next addresses: " + ', '.join(hex(addr) for addr in block['out_edge_list']) + "\n\n"
    
    return asm_str

def collect_feature_impact_on_prediction(data, model, label, top_k=5):
    """
    逐个将data.x中的基本块特征依次置零, 观察模型预测结果 并输出到文件中。
    
    参数:
    - data: 包含x成员的图数据对象。
    - model: 要评估的模型。
    - file_path: 保存结果的文件路径。
    - label: 模型的标签 0是良性、1恶意
    - top_k: 默认只输出前5个影响最大的
    
    返回值:
    - impact_str, 各个特征清零之后影响的的概率
    - top_k_index_list, 前top_k个特征序号
    """
    model.eval()
    changes = []
    
    impact_str = ""
    
    # 原始模型预测
    with torch.no_grad():
        out_init,_ = model(data)
        out_init = torch.exp(out_init)
        init_probs = out_init.tolist()  # 将tensor转换为list，便于写入文件

    # 逐个特征置零并评估影响
    for i in range(data.x.size(0)):
        data_copy = copy.deepcopy(data)
        data_copy.x[i] = torch.zeros(data_copy.x.shape[1])
        with torch.no_grad():
            out_modified,_ = model(data_copy)
            out_modified = torch.exp(out_modified)
            modified_probs = out_modified.tolist()
            changes.append((i, modified_probs[0][label]))  # 存储每次变化的结果,只保存0类别的
    # 将变化写入文件
    changes.sort(key=lambda x: x[1], reverse=False)

    impact_str += f"Init probability probabilities: {init_probs}\n\n"
    top_k_index_list = []
    for index, probs in changes:
        top_k -= 1
        impact_str += f"Feature {index} zeroed out, probabilities: {probs}\n"
        top_k_index_list.append(index)
        if top_k == 0:
            break
    
    return impact_str+"\n", top_k_index_list

def build_output_dir(dir):
    
    folder_path = os.path.join(dir, "bb_impact")
    print(folder_path)
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 文件夹不存在，创建它
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        # 文件夹已存在
        print(f"Folder already exists: {folder_path}")

def output_file(all_str, output_path):
    """按照原始文件格式输出内容到文件"""
    with open(output_path, 'w') as file:
        file.write(all_str)
    file.close()

device = torch.device("cpu")
model = DGCNN(num_features=20, num_classes=2)
model.load_state_dict(
    torch.load("/home/lebron/IRattack/py/model/record/dgcnn_2.pth", map_location=device))
model = model.to(device)
model.eval()

data_dir = "/home/lebron/disassemble"
cfg_dir = f"{data_dir}/cfg"
dataset = CFGDataset(root=data_dir)
data_loader = DataLoader(dataset, batch_size=1)
build_output_dir(dir=data_dir)
bb_impact_dir = f"{data_dir}/bb_impact"

idx = 0
for data in tqdm.tqdm(data_loader):
    cfg_name = dataset.raw_file_names_list[idx]
    out, top_k_indices = model(data)
    predictions = out.argmax(dim=1).tolist()[0] # 0良性 or 1恶意
    cfg = json.load( open(f"{cfg_dir}/{cfg_name}",'r'))
    
    feature_impact_str, top_k_index = collect_feature_impact_on_prediction(data, model, predictions)
    asm_str = format_asm(cfg=cfg, block_k_index=top_k_index)
    
    output_file(feature_impact_str+asm_str, f"{bb_impact_dir}/{cfg_name[:-5]}")
    
    idx += 1


    
    






