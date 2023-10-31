from typing import List
import numpy as np
import random


class Point(object):
    def __init__(self, x, y, type=None, valid=None) -> None:
        self.x = x
        self.y = y
        self.type = type
        self.valid = valid
    
    def __str__(self) -> str:
        return f'Point[{self.x}-{self.y}]'
            
class Points(object):
    def __init__(self, val: List[Point] = []) -> None:
        self.points = val
    def add_point(self, point: Point, center: Point) -> None:
        pt = Point(x=point.x - center.x, y=point.y - center.y, type=point.type, valid=point.valid)
        self.points.append(pt)

    def __getitem__(self, idx):
        return self.points[idx]
    def __len__(self):
        return len(self.points)
    def __iter__(self):
        if not self.points:
            return
        for d in self.points:
            yield d
    def __str__(self) -> str:
        all_pts = [f'{i.x}-{i.y}' for i in self.points]
        if len(self.points)==0:
            all_pts = ['empty']
        return ','.join(all_pts)

class Grid(object):
    def __init__(
        self,
        index_x: int, # 当前grid的index
        index_y: int,
        grid_size: float,
        point_limit: int, # grid内最多保存数量
        points: Points = Points([]), 
    ) -> None:
        
        self.index_x = index_x
        self.index_y = index_y
        
        self.grid_size = grid_size
        
        self.point_limit = point_limit

        self.center_x = index_x * grid_w + grid_w/2
        self.center_y = index_y * grid_h + grid_h/2
        self.center_point = Point(self.center_x, self.center_y)

        self.points: Points = points
        
    def valid_point_num(self):
        n = len(self.points)
        if n > self.point_limit:
            return self.point_limit
        return self.point_num()

    def point_num(self):
        return len(self.points)
    
    def add_point(self, point: Point):
        self.points.add_point(point=point, center=self.center_point)

    def get_feature(self, shuffle: bool = False) -> np.ndarray:
        """
        chw
        h=1
        """
        feature = np.zeros((4, 1, self.point_limit), dtype=np.float32)
        points = self.points.points
        if not points:
            return feature

        _size = min(self.point_num(), self.point_limit)

        index = [i for i in range(self.point_num())]
        if shuffle:
            random.shuffle(index)

        for i in range(_size):
            idx = index[i]
            feature[0, 0, i] = points[idx].x
            feature[1, 0, i] = points[idx].y
            feature[2, 0, i] = points[idx].type
            feature[3, 0, i] = points[idx].valid

        feature[0:2, 0, :] /= self.grid_size
        feature[2, 0, :] /= 10.0

        return feature


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
                _temp.append(Points([]))
            self._grid.append(_temp)
    
    def map2index(self, pts: Point) -> List:
        h_idx = int(pts.y / self.grid_h)
        w_idx = int(pts.x / self.grid_w)
        return h_idx, w_idx
         
    def add2grid(self, pts: Point) -> None:
        _hidx, _widx = self.map2index(pts)
        # print(_hidx, _widx)
        self._grid[_hidx][_widx].add_pts(pts)
    
    def 
    
    def show_grid(self) -> None:
        for i in self._grid:
            for j in i:
                print(j)

if __name__ == '__main__':
    demo = Grid(10, 10, 5, 5)
    demo.add2grid(Point(x=6, y=2))
    demo.add2grid(Point(x=9, y=2))
    demo.add2grid(Point(x=1, y=2))
    demo.add2grid(Point(x=6, y=6))
    
    demo.show_grid()
    