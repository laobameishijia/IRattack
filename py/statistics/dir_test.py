import copy
import json
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from model.src.gnn import run_measure
from model.src.dataset_construct import run_disassemble
from model.src.dataset_construct import run_extract_cfg
from fuzz_attack import disassemble, extract_cfg, measure
from model.src.gnn.model import DGCNN
from torch_geometric.data import Data




device = torch.device("cpu")
# model = DGCNN(num_features=9, num_classes=2)
# model.load_state_dict(
#     torch.load("/home/lebron/IRattack/py/model/record/semantics_dgcnn_2.pth", map_location=device))
# model = model.to(device)
# model.eval()

# 对单个目录做测试
data_dir="/home/lebron/IRFuzz/ELF/350"
disassemble(fuzz_dir=data_dir)
extract_cfg(fuzz_dir=data_dir)
data, result, predictions, _= measure(fuzz_dir=data_dir, model="dgcnn")
result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)

probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本

"""

dgcnn:
    check_O0:
    probability_0: 0.0532820001244545, probability_1: 0.9467179775238037

    check_bcf_O0
    probability_0: 0.0, probability_1: 1.0
    
    check_fla
    probability_0: 0.9913820028305054, probability_1: 0.008617999963462353


semantics_dgcnn:

    check_O0:
    probability_0: 0.5283820033073425, probability_1: 0.47161799669265747

    check_bcf_O0
    probability_0: 0.006519999820739031, probability_1: 0.9934800267219543
    
    check_fla
    probability_0: 0.9880070090293884, probability_1: 0.011993000283837318


"""