import argparse
import torch
import torch.nn as nn
import tqdm
import numpy as np

from sklearn import metrics
from torch_geometric.loader import DataLoader


from model.src.gnn.model import GIN0WithJK, GIN0, DGCNN
from model.src.gnn.dataset import CFGDataset_Semantics_Preseving, CFGDataset_MAGIC,CFGDataset_MAGIC_Attack


def split_pred_label(predictions, labels):
    target_class = 0

    sliced_predictions = []
    sliced_labels = []
    slice_index = 0

    for idx, label in enumerate(labels):
        if label != target_class:
            sliced_labels.append(labels[slice_index:idx])
            sliced_predictions.append(predictions[slice_index:idx])
            slice_index = idx
            target_class += 1
        
        if idx == len(labels) - 1:
            sliced_labels.append(labels[slice_index:])
            sliced_predictions.append(predictions[slice_index:])
    
    return sliced_predictions, sliced_labels


def measure(data_dir, model_name):
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_size = model_name.split("_")[1]
    model_type = model_name.split("_")[0]
    if feature_size == '9':
        dataset = CFGDataset_Semantics_Preseving(root= data_dir)
    elif feature_size == '20':
        dataset = CFGDataset_MAGIC_Attack(root= data_dir)
    else: 
        raise NotImplementedError
    
    if model_type == "DGCNN":
        model = DGCNN(num_features=int(feature_size), num_classes=dataset.num_classes)
    elif model_type == "GIN0":
        model = GIN0(num_features=int(feature_size), num_layers=4, hidden=64,num_classes=dataset.num_classes)
    elif model_type == "GIN0WithJK":
        model = GIN0WithJK(num_features=int(feature_size), num_layers=4, hidden=64,num_classes=dataset.num_classes)
    else:
        print(f"Model: {model_type} is not exist!")
        raise NotImplementedError
    
    model.load_state_dict(
        torch.load(f"/root/IRattack/py/model/record/{model_name}.pth", map_location=device))
    
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=5)
    model = model.to(device)
    model.eval()
    
    predictions = []
    labels = []
    for data in tqdm.tqdm(val_loader):
        data = data.to(device)
        if model_type == "DGCNN":
            out,top_k_indices = model(data)
        else:
            out = model(data)
            top_k_indices = -1
        # print(out)
        result = torch.exp(out) # 将模型输出的logsoftmax转换为softmax
        formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)
        probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
        probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
        print(f"probability_0: {probability_0}, probability_1: {probability_1}")
        
        predictions.extend(out.argmax(dim=1).tolist())
        labels.extend(data.y.tolist())
        print(f"predictions: {predictions}")
        print(f"label: {labels}")

    return data,out,predictions,top_k_indices

if __name__ == '__main__':
    data_dir="/home/lebron/disassemble/attack/"
    measure(data_dir)
    exit()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='data dir')
    args = parser.parse_args()
    data_dir = args.data_dir
    result_file_path = data_dir + "/result.txt"
    result_file = open(result_file_path, 'w')
    
    # big2015_dataset = CFGDataset_Normalized_After_BERT(
    #     root='/home/wubolun/data/malware/big2015/further',
    #     vocab_path='/home/wubolun/data/malware/big2015/further/set_0.5_pair_30/normal.vocab',
    #     seq_len=64)

    # dataset = CFGDataset_MAGIC_Attack(root='/home/lebron/disassemble')
    dataset = CFGDataset_MAGIC_Attack(root= data_dir)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=5)
    model = DGCNN(num_features=20, num_classes=dataset.num_classes)
    model.load_state_dict(
        torch.load("/home/lebron/MCFG_GNN/record/dgcnn_2.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    predictions = []
    labels = []
    for data in tqdm.tqdm(val_loader):
        data = data.to(device)
        out = model(data)
        print(out)
        predictions.extend(out.argmax(dim=1).tolist())
        labels.extend(data.y.tolist())
        print(f"predictions: {predictions}")
        result_file.write(f"predictions: {out.argmax(dim=1).tolist()[0]}\n")
        print(f"label: {labels}")
    result_file.close()
    exit()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # big2015_dataset = CFGDataset_Normalized_After_BERT(
    #     root='/home/wubolun/data/malware/big2015/further',
    #     vocab_path='/home/wubolun/data/malware/big2015/further/set_0.5_pair_30/normal.vocab',
    #     seq_len=64)

    big2015_dataset = CFGDataset_MAGIC(root='/home/wubolun/data/malware/big2015/further')

    recalls = []
    precisions = []
    f1s = []

    for k in range(5):

        train_idx, val_idx = big2015_dataset.train_val_split(k)
        val_dataset = big2015_dataset[val_idx]
        val_loader = DataLoader(val_dataset, batch_size=18, shuffle=False, num_workers=5)

        # model = DGCNN(num_features=20, num_classes=big2015_dataset.num_classes)
        model = GIN0WithJK(num_features=20, num_layers=5, hidden=128, num_classes=big2015_dataset.num_classes)
        model.load_state_dict(
            torch.load('result/magic-gin0jk-5-128/gin0jk_{}.pth'.format(k), map_location='cuda:0'))
        model = model.to(device)
        model.eval()

        # criterion = nn.NLLLoss()

        predictions = []
        labels = []

        running_loss = 0.0
        for data in tqdm.tqdm(val_loader):
            data = data.to(device)
            out = model(data)

            predictions.extend(out.argmax(dim=1).tolist())
            labels.extend(data.y.tolist())
            # loss = criterion(out, data.y)
            # running_loss += loss.item() * data.y.size(0)

        # print('k: {}, loss: {}'.format(k, running_loss/len(val_loader.dataset)))

        recall = list(metrics.recall_score(labels, predictions, average=None))
        precision = list(metrics.precision_score(labels, predictions, average=None))
        f1 = list(metrics.f1_score(labels, predictions, average=None))

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    # average on 5-fold
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    f1s = np.array(f1s)

    mean_recall = list(np.mean(recalls, 0))
    mean_precision = list(np.mean(precisions, 0))
    mean_f1 = list(np.mean(f1s, 0))

    print('recall: {}'.format(mean_recall))
    print('precision: {}'.format(mean_precision))
    print('f1 score: {}'.format(mean_f1))

    print('average recall: {}'.format(np.mean(np.array(mean_recall))))
    print('average precision: {}'.format(np.mean(np.array(mean_precision))))
    print('average f1 score: {}'.format(np.mean(np.array(mean_f1))))