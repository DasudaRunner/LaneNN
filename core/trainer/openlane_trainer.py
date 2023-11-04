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

from core.trainer.base_trainer import BaseTrainer
from core.data.build_dataset import build_dataset
from core.models.build_model import build_model
from utils.util import AverageMeter

class OpenLaneTrainer(BaseTrainer):
    def __init__(self, C):
        super().__init__(C)
        self.C = C

        self.create_dataloader()
        self.create_model()
        self.create_loss()
        self.create_optimizer()

    def create_dataloader(self):
        config = self.C

        self.epoch = config['common']['epoch']
        self.batch_size = config['common']['bs']
        
        # add grid config in dataset config
        dataset_cfg = config['dataset']
        dataset_cfg['kwargs'].update(config['grid'])
        
        self.dataset = build_dataset(dataset_cfg)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.batch_size, 
                                     shuffle=True,
                                     num_workers=config['common']['works'])
    
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
            
            self.optimizer.step()
            
            tmp.batch_time.update(time.time() - end)
            tmp.loss_meter.update(tmp.loss.item())
            
            now_date = datetime.now().strftime("%m-%d(%H:%M:%S)")
            
            print('{0}: [{1}], '
                'BT {batch_time.avg:.3f}({data_time.avg:.3f}), '
                'loss {loss.val:.3f}'.format(
                now_date, idx,
                batch_time=tmp.batch_time, data_time=tmp.data_time, 
                loss=tmp.loss_meter))
            sys.stdout.flush()

            end = time.time()