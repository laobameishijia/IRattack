# created by Wu Bolun
# 2020.3.1
# bowenwu@sjtu.edu.cn

import argparse
import copy
import sys

import torch
import torch.nn as nn
import tqdm
from torch_geometric.data import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel

# local import
sys.path.append('..')
# from bert_tidy.vocab import *

from dataset import (BBSDataset, CFGDataset_MAGIC, CFGDataset_Normalized,
                     CFGDataset_Normalized_After_BERT)
from model import DGCNN, GIN0, GIN0WithJK


class GNNTrainer(object):
    def __init__(self, gnn, train_loader, val_loader, 
                 optimizer, criterion, scheduler=None, name='dgcnn', multi_gpu=0):

        self.multi_gpu = multi_gpu
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        if self.multi_gpu:
            self.model = DataParallel(gnn)
        else:
            self.model = gnn
        self.model = self.model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_size = len(self.train_loader.dataset)
        self.val_size = len(self.val_loader.dataset)

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.name = name

        self.best_model_dict = copy.deepcopy(self.model.state_dict())
        self.best_val_acc = 0.0
    
    def train(self, num_epochs):
        self.num_epochs = num_epochs
        self.step = 0

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            train_loss, train_acc = self._train()
            val_loss, val_acc = self._val()

            print('Epoch: {}'.format(epoch))
            print('Train Loss: {:.4f}, Train Acc: {:.4f}'.format(train_loss, train_acc))
            print('Val Loss: {:.4f}, Val Acc: {:.4f}'.format(val_loss, val_acc))

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_dict = copy.deepcopy(self.model.state_dict())
                self.save_model()
        
        print('Best val acc: {:.4f}'.format(self.best_val_acc))
        with open('record/best_val_acc.txt', 'a') as f:
            f.write('{}: {}\n'.format(self.name, self.best_val_acc))
                
    def save_model(self):
        torch.save(self.best_model_dict, './record/{}.pth'.format(self.name))

    def _train(self):
        ''' one epoch train '''
        self.model.train()

        running_loss = 0.0
        running_corrects = 0

        iter_count = 0
        for data in self.train_loader:

            if self.multi_gpu:
                out = self.model(data)
                y = torch.cat([d.y for d in data]).to(out.device)
            else:
                data = data.to(self.device)
                out = self.model(data)
                y = data.y

            preds = out.argmax(dim=1)
            loss = self.criterion(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * y.size(0)
            correct_count = int((preds == y).sum())
            running_corrects += correct_count
            
            print('Step {}, Epoch {}/{}, iter {}/{}, Loss {:.4f}, Acc {:.4f}.'.format(
                self.step, self.epoch, self.num_epochs-1,
                iter_count, int(self.train_size/CFG_BATCH_SIZE),
                loss.item(), correct_count/float(data.y.size(0))))
        
            iter_count += 1
            self.step += 1

        # end one epoch
        if self.scheduler is not None:
            self.scheduler.step()

        return running_loss/self.train_size, running_corrects/self.train_size
    
    def _val(self):
        self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        for data in self.val_loader:
            if self.multi_gpu:
                out = self.model(data)
                y = torch.cat([d.y for d in data]).to(out.device)
            else:
                data = data.to(self.device)
                out = self.model(data)
                y = data.y

            preds = out.argmax(dim=1)
            loss = self.criterion(out, y)

            running_loss += loss.item() * y.size(0)
            running_corrects += int((preds == y).sum())

        return running_loss/self.val_size, running_corrects/self.val_size


if __name__ == '__main__':

    CFG_BATCH_SIZE = 32
    NUM_WORKERS = 5
    FEATURE_SIZE = 20  # 这个特征是特征组成的特征 MAGIC的特征是20维度
    # data
    # big2015_dataset = CFGDataset_MAGIC(root='F:\研二上\论文\dataset')
    big2015_dataset = CFGDataset_MAGIC(root='F:\研二上\论文\dataset')
    
    
    # 5-fold train&val
    for k in range(5):

        # specific
        if k != 2:
            continue

        print('K: {}'.format(k))
        train_idx, val_idx = big2015_dataset.train_val_split(k)
        train_dataset = big2015_dataset[train_idx]
        val_dataset = big2015_dataset[val_idx]
        print('Data loaded: train {}, eval {}.'.format(len(train_dataset), len(val_dataset)))

       
        train_loader = DataLoader(train_dataset, batch_size=CFG_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=CFG_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # GNN
        model = DGCNN(num_features=FEATURE_SIZE,
                        num_classes=big2015_dataset.num_classes)
        print('Model: {}'.format("DGCNN"))

        # loss and optim
        criterion = nn.NLLLoss()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        print('Starting training...')

        trainer = GNNTrainer(gnn=model, train_loader=train_loader, val_loader=val_loader, 
            optimizer=optimizer, criterion=criterion, scheduler=scheduler, 
            name='{}_{}'.format("dgcnn", k), multi_gpu=0)
        trainer.train(num_epochs=70)

    exit()