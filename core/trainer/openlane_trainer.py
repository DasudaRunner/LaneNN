import os
import os.path as osp
import math
import torch
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import pdb
import time

import shutil
import random

from torch.utils.data import DataLoader

from core.trainer.base_trainer import BaseTrainer
from core.data.build_dataset import build_dataset
from utils.util import AverageMeter

class OpenLaneTrainer(BaseTrainer):
    def __init__(self, C):
        super().__init__(C)
        self.C = C

        self.create_dataloader()
        # self.create_model()
        # self.create_loss()
        # self.create_optimizer()

    def create_dataloader(self):
        config = self.C

        self.total_iter = config['common']['max_iters']
        self.batch_size = config['common']['bs']
        
        self.dataset = build_dataset(config['dataset'])
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.batch_size, 
                                     shuffle=True,
                                     num_workers=config['common']['works'])
    
    def create_model(self):
        pass

    def create_loss(self):
        pass
    
    def create_optimizer(self):
        pass

    def pre_run(self):
        tmp = self.tmp
        
        tmp.batch_time = AverageMeter()
        tmp.data_time = AverageMeter()
        tmp.loss_meter = AverageMeter()
        tmp.acc_meter = AverageMeter()

    def run(self):
        config = self.config
        tmp = self.tmp
        for idx, tmp.input in enumerate(self.dataloader):
            feat = tmp.input['feat']
            label = tmp.input['label']
            pdb.set_trace()
        