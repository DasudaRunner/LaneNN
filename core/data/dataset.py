import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import List

class PointNetDataset(Dataset):
    def __init__(
        self,
        index_path: str,
        point_num=64,
        preload: bool = False,
        shuffle_in_grid: bool = False,
        add_center_lane: bool = False,
        trans = [],
        sample_stride: float = 2.0,
        sample_num_per_lane: int = 100,
        sample_lane_num: int = 8,
    ):
        self.sample_lane_num = sample_lane_num
        self.sample_num_per_lane = sample_num_per_lane
        self.sample_stride = sample_stride
        self.add_center_lane = add_center_lane
        self.trans = trans
        self.shuffle_in_grid = shuffle_in_grid
        self.point_num = point_num
        self.index_path = index_path
        self.size: int = 0
        self.labels: List[int] = []
        # self.transformers: List[LaneTransform] = transformers
        self.lines = []
        self.cs = []
        self.preload = preload
        self.preload_data: dict = {}
        self._load()

    def _load_item(self, path, from_disk: bool = True) -> Feature:
        if from_disk:
            with open(os.path.join(path, "frame.pkl"), "rb") as f:
                frame_v2: V2Frame = pickle.load(f)

                feature_v4 = Feature(
                    pose=frame_v2.pose,
                    lanes=frame_v2.lanes,
                    sample_lane_k=self.sample_lane_num,
                    sample_stride=self.sample_stride,
                    grid_size=grid_size,
                    grid_x_num=grid_height,
                    grid_y_num=grid_width,
                    grid_point_limit=grid_point_num,
                    add_center_lane=self.add_center_lane,
                    sample_num_per_lane=self.sample_num_per_lane,
                )

                return feature_v4
        else:
            return self.preload_data[path]

    def _load(self):
        with open(self.index_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip("\n") for line in lines]
            self.size = len(lines)
            self.lines = lines
            self.cs = [line.split(" ") for line in lines]

        if self.preload:
            for cs in self.cs:
                path, _ = cs
                feature = self._load_item(path=path, from_disk=True)
                self.preload_data[path] = feature

    def _transform(self, feat: Feature) -> Feature:
        for trans in self.trans:
            feat = trans.process(feat)
        return feat

    def _to_tensor_2d(self, feature: Feature, unsqueeze: bool = False) -> torch.Tensor:
        blob = feature.gen_feature(shuffle=self.shuffle_in_grid)

        ret = torch.from_numpy(blob)

        if unsqueeze:
            return ret.unsqueeze(0)
        return ret

    def __getitem__(self, index):
        path, label = self.cs[index]
        label_v = int(label)
        label = np.array(label_v)

        feature = self._load_item(path, from_disk=False if self.preload else True)
        feature.set_label(label_v)

        feature = self._transform(feature)
        feature = self._to_tensor_2d(feature=feature)
        return feature, label

    def __len__(self):
        return self.size
