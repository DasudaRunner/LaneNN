from typing import Optional, Sized
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler

class BanlanceSampler(Sampler):
    def __init__(self, dataset, batch_size, imageNumPerClass, epoch) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.imageNumPerClass = imageNumPerClass
        self.epoch = epoch
        self.max_sample = epoch * len(dataset)
        
        self.indices = self.gen_indices()

    def __iter__(self):
        return iter(self.indices)

    def gen_indices(self):
        class2id = defaultdict(list)
        for idx, i in enumerate(self.dataset.all_meta):
            class2id[i].append(idx)
            
        max_class_num = 0
        for v in class2id.values():
            max_class_num = max(len(v), max_class_num)
        all_keys = list(class2id.keys())
        for i in range(max_class_num):
            for _c in all_keys:
                self.indices.append(class2id[_c][i%len(class2id[_c])])
        
        
        
