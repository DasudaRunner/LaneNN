from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

# 示例数据 - 曲线点
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 3, 1, 2, 3, 4])

plt.scatter(x, y)


# 使用样条插值拟合曲线
cs = CubicSpline(x, y)

# 定义采样点数量
num_samples = 10

# 在参数空间均匀采样点
t = np.linspace(0, 1, num_samples)

# 获取采样后的 x 和 y 坐标
sampled_x = cs(t)
sampled_y = cs(t)

print(sampled_x)
print(sampled_y)


plt.show()