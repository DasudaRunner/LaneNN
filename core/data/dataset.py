import numpy as np
import torch
from torch.utils.data import Dataset
from utils.misc import *
import os
import os.path as osp
from typing import List

class OpenLaneDataset(Dataset):
    def __init__(self, config):
        self.cfg = config
        
        self.data_list = config['data_list']
        self.data_prefix = config.get('data_prefix', '')
        self.max_points_in_grid = config['max_points_in_grid']
        
        self.all_data = self._preload()
        self.data_len = len(self.all_data)
        
    def _preload(self):
        all_data = []
        _data = read_list(self.data_list)
        if self.data_prefix:
            _data = [osp.join(self.data_prefix, i) for i in _data]
        for _val in _data:
            all_data.append(self.parse_openlane(_val))
        return all_data
        
    def parse_openlane(self, json_file: str) -> List:
        _json = load_json(json_file)
        all_lines = _json['lane_lines']
        new_lines = []
        all_category = []
        for idx, sline in enumerate(all_lines):
            category = sline['category']
            coord = sline['uv'] # 2*n
            all_category.append*(category)
            new_lines.append({'category': category, 'coord': coord})
        # TODO 
        # 生成新的label
        gt = 0
        if 20 in all_category:
            gt = 1
        
        return new_lines
    
    def __getitem__(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.data_len)
        path, label = self.all_data[idx]
        
        label_v = int(label)
        label = np.array(label_v)

        feature = self._load_item(path, from_disk=False if self.preload else True)
        feature.set_label(label_v)

        feature = self._transform(feature)
        feature = self._to_tensor_2d(feature=feature)
        return feature, label

    def __len__(self):
        return self.size
