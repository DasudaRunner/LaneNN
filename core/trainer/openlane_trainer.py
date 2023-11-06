import os, sys
import os.path as osp
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pdb
import time
from datetime import datetime

import shutil
import random

from torch.utils.data import DataLoader
from tqdm import tqdm

from core.trainer.base_trainer import BaseTrainer
from core.data.build_dataset import build_dataset
from core.models.build_model import build_model
from utils.util import AverageMeter
from utils.nn import accuracy

class OpenLaneTrainer(BaseTrainer):
    def __init__(self, C):
        super().__init__(C)
        

        if 'eval' not in self.config:
            self.create_dataloader()
            self.create_model()
            self.create_loss()
            self.create_optimizer()
        else:
            self.create_test_dataloader()
            self.create_model()
            self.load()

    def create_dataloader(self):
        config = self.config

        self.epoch = config['common']['epoch']
        self.batch_size = config['common']['bs']
        
        # add grid config in dataset config
        dataset_cfg = config['dataset']
        dataset_cfg['kwargs'].update(config['grid'])
        
        self.dataset = build_dataset(dataset_cfg)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.batch_size, 
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=config['common']['works'])
    def create_test_dataloader(self):
        config = self.config

        # add grid config in dataset config
        dataset_cfg = config['test_dataset']
        dataset_cfg['kwargs'].update(config['grid'])
        
        self.dataset = build_dataset(dataset_cfg)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=4, 
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=0)

 
    def create_model(self):
        config = self.config
        model_cfg = config['model']
        model_cfg['kwargs'].update(config['grid'])
        self.model = build_model(model_cfg)

    def create_loss(self):
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.CrossEntropyLoss()
    
    def create_optimizer(self):
        config = self.config
        self.init_lr = config['common']['lr']
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.9, weight_decay=1e-4)

    def pre_run(self):
        tmp = self.tmp
        
        tmp.batch_time = AverageMeter()
        tmp.data_time = AverageMeter()
        tmp.loss_meter = AverageMeter()
        tmp.acc_meter = AverageMeter()

    def save(self, idx):
        # opt_state_dict = self.optimizer.state_dict()
        model_state_dict = self.model.state_dict()
        
        state = {
            'model': model_state_dict,
            # 'optimizer': opt_state_dict,
        }
        torch.save(state, osp.join(self.ckpt_path, f'model_{idx}.pth'))
    
    def load(self):
        config = self.config
        checkpoint = torch.load(config['load_path'], map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
    
    def eval(self):
        self.model.eval()
        all_pred = []
        all_gt = []
        for idx, input in tqdm(enumerate(self.dataloader)): 
            feat = input['feat'].float()
            label = input['label'].long()
            with torch.no_grad():
                out = self.model(feat)
                all_pred.append(out['pred'])
                all_gt.append(label)
        
        all_pred = torch.concatenate(all_pred, dim=0)
        all_gt = torch.concatenate(all_gt, dim=0)
        
        print(all_pred)
        
        res = accuracy(all_pred, all_gt)[0]
        print(res.item())
    
    def run(self):
        config = self.config
        tmp = self.tmp

        self.pre_run()
        
        self.model.train()
        end = time.time()
        print('Start training...')
        for idx, tmp.input in enumerate(self.dataloader): 
            tmp.data_time.update(time.time() - end)   
            self.optimizer.zero_grad()
            
            feat = tmp.input['feat'].float()
            label = tmp.input['label'].long()
            
            out = self.model(feat)
            # print(out['pred'].shape)
            # print(label.shape)
            # pdb.set_trace()
            
            tmp.loss = self.loss(out['pred'], label)
            tmp.loss.backward()
            
            tmp.acc = accuracy(out['pred'], label)[0]
            
            self.optimizer.step()
            
            tmp.batch_time.update(time.time() - end)
            tmp.loss_meter.update(tmp.loss.item())
            tmp.acc_meter.update(tmp.acc.item())
            
            now_date = datetime.now().strftime("%m-%d(%H:%M:%S)")
            
            print('{0}: [{1}], '
                'BT {batch_time.avg:.3f}({data_time.avg:.3f}), '
                'loss {loss.avg:.3f}({acc.avg:.3f})'.format(
                now_date, idx,
                batch_time=tmp.batch_time, data_time=tmp.data_time, 
                loss=tmp.loss_meter, acc=tmp.acc_meter))
            sys.stdout.flush()

            end = time.time()

            if idx%10==0:
                self.save(idx)
                
        self.save('final')