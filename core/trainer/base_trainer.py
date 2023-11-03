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
import gc

from easydict import EasyDict as edict

import shutil
import random

class WorkerInit(object):
    def __init__(self, num_workers):
        self.num_workers = num_workers
    def func(self, pid):
        # print('setting worker seed {}'.format(self.rank*self.num_workers+pid), flush=True)
        np.random.seed(self.num_workers+pid)
        random.seed(self.num_workers+pid)

# 非分布式训练
class BaseTrainer(object):
    """class for BaseTrainer"""
    def __init__(self, C):
        self.config = C

        self.mixPrecision = self.config['common'].get('mixPrecision', 'fp16')
        self.init_path()
        self.set_seed()

        # freq
        self.print_freq = self.config['common'].get('print_freq', 10)
        self.save_ckpt_freq = self.config['common'].get('save_freq', 10000)
        self.save_ckpt_newest_freq = self.config['common'].get('save_newest_freq', 1000)
        self.tb_freq = self.config['common'].get('tb_freq', 100)

        self.last_iter = -1
        
        self.tmp = edict()

    def set_seed(self):
        config = self.config
        seed = config['common']['seed']
        works = config['common']['works']
        
        worker_init = WorkerInit(works)
        self.worker_init_fn = worker_init.func
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def create_model(self):
        pass

    def create_dataloader(self):
        pass
    
    def create_loss(self):
        pass
    
    def create_optimizer(self):
        pass

    def init_path(self):
        config = self.config
        
        save_path = config.get('save_path', os.path.dirname(self.config['config_file']))
        event_path = osp.join(save_path, 'events')
        ckpt_path = osp.join(save_path, 'checkpoints')

        if not os.path.exists(event_path):
            os.makedirs(event_path)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        shutil.copy(self.config['config_file'], ckpt_path)

        self.tb_logger = SummaryWriter(log_dir=event_path, flush_secs=10)

        self.ckpt_path = ckpt_path