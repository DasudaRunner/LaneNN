from typing import List

# record index of point
class Index(object):
    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j

    def value(self):
        return (self.i, self.j)

class Point(object):
    def __init__(self, x, y, type=None, valid=None) -> None:
        self.x = x
        self.y = y
        self.type = type
        self.valid = valid

    def trans(self, offset: "Point"):
        self.x -= offset.x
        self.y -= offset.y

    # @classmethod
    # def gen_empty_point(cls):
    #     return Point(x=0, y=0, type=LaneType.unknown, valid=PointValidType.not_valid)

class Points(object):
    def __init__(self, points: List[Point] = []) -> None:
        self.points = points

    def add_point(self, center: Point, point: Point):
        p = Point(x=point.x - center.x, y=point.y - center.y, type=point.type, valid=point.valid)
        self.points.append(p)

    def __getitem__(self, index):
        return self.points[index]

    def __len__(self):
        return len(self.points) if self.points else 0

    def __iter__(self):
        if not self.points:
            return
        for d in self.points:
            yield d

class Grid(object):
    def __init__(
        self,
        index_x: int,
        index_y: int,
        x_num: int,
        y_num: int,
        grid_size: float,
        point_limit: int,
        points: Points = Points([]),
    ) -> None:
        self.id = "{}_{}".format(index_x, index_y)
        self.index_x = index_x
        self.index_y = index_y
        self.grid_size = grid_size
        self.point_limit = point_limit

        self.center_x = index_x * grid_size
        self.center_y = (index_y - y_num // 2) * grid_size
        self.center_point = Point(self.center_x, self.center_y)

        self.points: Points = points

    def valid_point_num(self):
        n = len(self.points)
        if n > self.point_limit:
            return self.point_limit
        return self.point_num()

    def point_num(self):
        return len(self.points)

    def is_valid(self):
        return len(self.points) > 0

    def add_point(self, point: Point):
        self.points.add_point(point=point, center=self.center_point)

    def get_feature(self, shuffle: bool = False) -> np.ndarray:
        """
        chw
        h=1
        """
        feature = np.zeros((4, 1, self.point_limit), dtype=np.float32)

        points: List[Point] = self.points.points
        if not points:
            return feature

        size = min(self.point_num(), self.point_limit)

        index = [i for i in range(self.point_num())]
        if shuffle:
            random.shuffle(index)

        for i in range(size):
            idx = index[i]
            feature[0, 0, i] = points[idx].x
            feature[1, 0, i] = points[idx].y
            feature[2, 0, i] = points[idx].type
            feature[3, 0, i] = points[idx].valid

        feature[0:2, 0, :] /= self.grid_size
        feature[2, 0, :] /= 10.0

        return feature

    def center(self):
        return Point(x=self.center_x, y=self.center_y)

    def to_global(self, point: Point):
        p = Point(x=point.x, y=point.y, type=point.type, valid=point.valid)
        p.x *= self.grid_size
        p.y *= self.grid_size

        p.x += self.center_x
        p.y += self.center_y

        return p


class Map(object):
    def __init__(
        self, grid_x_num: int, grid_y_num: int, grid_size: float, point_limit: int
    ) -> None:
        assert grid_y_num % 2 == 1
        assert grid_x_num > 0
        self.point_limit = point_limit

        self.x_num = grid_x_num
        self.y_num = grid_y_num
        self.grid_size = grid_size
        self.max_x = (grid_x_num - 0.5) * grid_size
        self.max_y = ((grid_y_num // 2) + 0.5) * grid_size
        self.min_x = -0.5 * grid_size
        self.min_y = -self.max_y

        self.memo: Dict[List, Grid] = {}
        for i in range(grid_x_num):
            for j in range(grid_y_num):
                self.memo[Index(i, j).value()] = Grid(
                    index_x=i,
                    index_y=j,
                    x_num=grid_x_num,
                    y_num=grid_y_num,
                    grid_size=grid_size,
                    point_limit=point_limit,
                    points=Points([]),
                )
        self.point_cnt = 0

    def get_feature(self, shuffle: bool = False) -> np.ndarray:
        """
        chw

        c=4
        h=m*n
        w=point_limit
        """
        cnt = 0
        for _, grid in self.memo.items():
            if cnt == 0:
                feature = grid.get_feature(shuffle)
            else:
                feature = np.concatenate((feature, grid.get_feature(shuffle)), axis=1)
            cnt += 1
        return feature

    def valid_grid_num(self):
        cnt = 0
        for _, grid in self.memo.items():
            if grid.is_valid():
                cnt += 1
        return cnt

    def valid_point_num(self):
        cnt = 0
        for _, grid in self.memo.items():
            cnt += grid.valid_point_num()
        return cnt

    def is_valid(self) -> bool:
        self.point_cnt > 0

    def get_index(self, x, y) -> Index:
        if x >= self.max_x or x <= self.min_x:
            return None
        if y >= self.max_y or y <= self.min_y:
            return None

        if x <= (0.5 * self.grid_size):
            x_index = 0
        else:
            x_index = 1 + math.floor((x - (0.5 * self.grid_size)) / self.grid_size)

        neg = False
        if y < 0:
            y = -y
            neg = True

        if y < 0.5 * self.grid_size:
            y_index = 0
        else:
            y_index = 1 + math.floor((y - (0.5 * self.grid_size)) / self.grid_size)

        if neg:
            y_index = -y_index

        y_index += self.y_num // 2

        return Index(x_index, y_index)

    def add_point(self, x: float, y: float, type: None, valid: None) -> bool:
        index = self.get_index(x, y)
        if not index:
            return False
        # print("x: {}, y: {}, index: {}".format(x, y, index.value()))
        grid = self.memo[index.value()]
        grid.add_point(Point(x, y, type, valid))
        self.point_cnt += 1
        return True

    def to_global(self, x_index, y_index, x: float, y: float) -> Point:
        grid = self.memo[Index(x_index, y_index).value()]
        p = grid.to_global(Point(x, y))
        return p

    def to_global_from_tensor(self, index, x: float, y: float) -> Point:
        x_index = index // self.y_num
        y_index = index % self.y_num
        return self.to_global(x_index, y_index, x, y)