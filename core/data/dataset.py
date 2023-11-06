import numpy as np
import torch
from torch.utils.data import Dataset
from utils.misc import *
from core.data.grid import Map
import os
import os.path as osp
from typing import List
from tqdm import tqdm
import pdb

from utils.my_math import _interp1d

class OpenLaneDataset(Dataset):
    def __init__(self, config):
        self.cfg = config
        
        self.data_list = config['data_list']
        self.data_prefix = config.get('data_prefix', '')
        self.max_points_in_grid = config['max_points_in_grid']
        
        self.all_meta = []
        self.all_data = self._preload()
        self.data_len = len(self.all_data)
        
    def _preload(self):
        all_data = []
        _data = read_list(self.data_list) # [:10]
        if self.data_prefix:
            _data = [osp.join(self.data_prefix, i) for i in _data]
        for _val in tqdm(_data):
            parse_res = self.parse_openlane(_val)
            if parse_res is not None:
                all_data.append(parse_res[0])
                self.all_meta.append(parse_res[1])
                
        return all_data
        
    def parse_openlane(self, json_file: str) -> List:
        def map_cat(old_cat: int) -> int:
            return old_cat if old_cat <= 12 else old_cat-7
        
        _json = load_json(json_file)
        all_lines = _json['lane_lines']
        new_lines = []
        all_category = []
        for idx, sline in enumerate(all_lines):
            category = sline['category']
            coord = sline['uv'] # 2*n
            single_line = np.array(coord)
            if single_line.shape[1] < 5:
                continue
            single_line[0, :] = single_line[0, :] / 1920 * 1280
            single_line[1, :] = single_line[1, :] / 1280 * 1280
            inter_line = _interp1d(single_line, inter_val=1.)
            new_lines.append(inter_line)
            
            new_cat = map_cat(category)
            all_category += [new_cat] * inter_line.shape[1]
        
        if len(new_lines) == 0:
            return None
   
        all_category = np.array(all_category) / 7. -1
        all_category = all_category.reshape((1, -1))
        new_lines = np.concatenate(new_lines, axis=1)
        new_lines = np.concatenate([new_lines, all_category], axis=0)
            
        # TODO 
        # 生成新的label
        gt = 0
        if len(all_lines) > 5:
            gt = 1    
        map = Map(128, 128, 10, self.max_points_in_grid)
        for _pi in range(new_lines.shape[1]):
            map.add_point(new_lines[0][_pi], new_lines[1][_pi], type=new_lines[2][_pi], valid=1.)
        return map, gt
    
    def __getitem__(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.data_len)
            
        _map = self.all_data[idx]
        label = self.all_meta[idx]
        
        out = {}
        feat = torch.from_numpy(_map.get_feature()).float()
        out['feat'] = feat
        out['label'] = label
        return out

    def __len__(self) -> int:
        return self.data_len
