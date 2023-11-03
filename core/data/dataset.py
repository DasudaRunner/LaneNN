import numpy as np
import torch
from torch.utils.data import Dataset
from utils.misc import *
from core.data.grid import Map
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
        _data = read_list(self.data_list)[:10]
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
            all_category.append(category)
            for pts_idx in range(len(coord[0])):
                ori_x = coord[0][pts_idx]
                ori_y = 1280-coord[1][pts_idx]
                norm_x = ori_x / 1920 * 1280
                norm_y = ori_y / 1280 * 1280
                new_lines.append([norm_x, norm_y, idx])
        # TODO 
        # 生成新的label
        gt = 0
        if 20 in all_category:
            gt = 1
        map = Map(128, 128, 10, 64)
        for d in new_lines:
            map.add_point(d[0], d[1], type=d[2], valid=1)
        return map, gt
    
    def __getitem__(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.data_len)
            
        _map, label = self.all_data[idx]
        
        out = {}
        feat = torch.from_numpy(_map.get_feature()).float()
        out['feat'] = feat
        out['label'] = label
        return out

    def __len__(self) -> int:
        return self.data_len
