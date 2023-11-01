from typing import List
import numpy as np
import random


class Point(object):
    def __init__(self, x, y, type=-1, valid=-1) -> None:
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

        self.center_x = index_x * grid_size + grid_size/2
        self.center_y = index_y * grid_size + grid_size/2
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

        feature[0:2, 0, :] /= self.grid_size / 2
        feature[2, 0, :] /= 10.0 # TODO

        return feature

class Map(object):
    def __init__(
        self, 
        grid_x_num: int, # w方向grid数量
        grid_y_num: int,  # h方向grid数量
        grid_size: float,
        point_limit: int
    ) -> None:
        assert grid_y_num > 0 and grid_x_num > 0

        self.x_num = grid_x_num
        self.y_num = grid_y_num
        self.grid_size = grid_size
        self.point_limit = point_limit

        self.max_x = (grid_x_num - 0.5) * grid_size
        self.max_y = ((grid_y_num // 2) + 0.5) * grid_size
        self.min_x = -0.5 * grid_size
        self.min_y = -self.max_y

        self.point_cnt = 0
        self.memo = {}
        for i in range(grid_x_num):
            for j in range(grid_y_num):
                self.memo[f'{i}-{j}'] = Grid(
                    index_x=i,
                    index_y=j,
                    grid_size=grid_size,
                    point_limit=point_limit,
                    points=Points([]),
                )

    def get_index(self, x, y) -> List:
        x_index = int(x / self.grid_size)
        y_index = int(y / self.grid_size)
        return x_index, y_index

    def add_point(self, 
                  x: float, 
                  y: float, 
                  type: None, 
                  valid: None) -> None:
        xidx, yidx = self.get_index(x, y)
        # print("x: {}, y: {}, index: {}".format(x, y, index.value()))
        grid = self.memo[f'{xidx}-{yidx}']
        grid.add_point(Point(x, y, type, valid))
        self.point_cnt += 1
    
    def get_feature(self, shuffle: bool = False) -> np.ndarray:
        """
        chw

        c=4
        h=m*n
        w=point_limit
        """
        lfeats = []
        for _, grid in self.memo.items():
            lfeats.append(grid.get_feature(shuffle))
        feature = np.concatenate(lfeats, axis=1)
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
    demo = Map(10, 10, 5, point_limit=2)
    demo.add_point(Point(x=6, y=2))
    demo.add_point(Point(x=9, y=2))
    demo.add_point(Point(x=1, y=2))
    demo.add_point(Point(x=6, y=6))
    
    res = demo.get_feature()
    print(res)
    