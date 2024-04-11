
import numpy as np
import torch
from model.src.gnn.model import DGCNN
from torch_geometric.data import Data
import matplotlib.pyplot as plt

def register_hooks(model):
    def forward_hook(module, input, output):
        print("Forward hook called.")
        module.output = output.detach()  # 记录前向传播的输出
    
    def backward_hook(module, grad_input, grad_output):
        print("Backward hook called.")
        print("Grad output:", grad_output[0])
        module.grad = grad_output[0].detach()  # 记录反向传播的梯度

    hook_handles = []
    hook_handles.append(model.conv4.register_forward_hook(forward_hook))
    hook_handles.append(model.conv4.register_backward_hook(backward_hook))
    return hook_handles

def grad_cam(model, data, target_class):
    # 注册钩子
    hooks = register_hooks(model)
    model.eval()  # 设置模型为评估模式

    # 前向传播
    output,_ = model(data)
    if output.ndim == 1:
        output = output.unsqueeze(0)  # 确保输出至少为2D
    target = output[:, target_class]
    target.backward()

    # 通过钩子访问输出和梯度
    activations = model.conv4.output  # 前向钩子记录的输出
    gradients = model.conv4.grad      # 反向钩子记录的梯度

    # 梯度池化
    pooled_gradients = torch.mean(gradients, dim=[0, 1])  # 在节点和特征维度上进行平均

    # 权重激活图层
    for i in range(activations.shape[1]):  # 遍历所有特征/通道
        activations[:, i] *= pooled_gradients.item()  # 使用.item()转换为数值

    # 生成类激活映射
    cam = torch.sum(activations, dim=1).cpu().numpy()  # 在特征维度上求和以生成CAM

    # 清除钩子
    for handle in hooks:
        handle.remove()

    return cam


def visualize_cam(cam):
    plt.figure(figsize=(10, 1))
    plt.imshow(cam[np.newaxis, :], cmap='hot', aspect='auto')
    plt.colorbar()
    plt.title('Class Activation Mapping')
    plt.show()


device = torch.device("cpu")
model = DGCNN(num_features=20, num_classes=2)
model.load_state_dict(
    torch.load("/home/lebron/IRattack/py/model/record/dgcnn_2.pth", map_location=device))
model = model.to(device)
# model.eval()

# 调用函数
data = torch.load("/home/lebron/disassemble/attack/cfg_magic_test/data_{}_{}.pt".format(1, "server"))
data = data.to(device)

cam_output = grad_cam(model, data, target_class=0)
print("Class Activation Map (CAM) Output Shape:", cam_output.shape)  # 输出形状验证
visualize_cam(cam_output)
exit()
