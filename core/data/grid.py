from typing import List


class Point(object):
    def __init__(self, x, y, type=None, valid=None) -> None:
        self.x = x
        self.y = y
        self.type = type
        self.valid = valid
        
class Points(object):
    def __init__(self, val: List[Point] = []) -> None:
        self.val = val
    def add_pts(self, pt: Point) -> None:
        self.val.append(pt)

    def __getitem__(self, idx):
        return self.val[idx]
    def __len__(self):
        return len(self.val)

class Grid(object):
    def __init__(self, h, w, grid_h, grid_w) -> None:
        self._grid = []
        self.grid_h = grid_h
        self.grid_w = grid_w
        grid_num_h = h // grid_h
        grid_num_w = w // grid_w
        # init grid param
        for i in range(grid_num_h):
            _temp = []
            for j in range(grid_num_w):
                _temp.append([])
            self._grid.append(_temp)
    
    def map2index(self, pts: Point) -> List:
        h_idx = int(pts.y / self.grid_h)
        w_idx = int(pts.x / self.grid_w)
    
    def add2grid(self, pts: Point) -> None:
        pass