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
        