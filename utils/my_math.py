import numpy as np
from scipy.interpolate import interp1d

def _interp1d(pts_list: np.ndarray, inter_val: float) -> np.ndarray:
    assert pts_list.shape[0] == 2
    x = pts_list[0, :]
    y = pts_list[1, :]
    new_x = np.linspace(x.min(), x.max(), int((x.max()-x.min())/inter_val))
    f = interp1d(x, y, kind='linear')
    new_y = f(new_x)
    return np.vstack((new_x, new_y))


# x= np.array([0, 1, 2, 3, 4, 5, 6, 7])
# y= np.array([3, 4, 3.5, 2, 1, 1.5, 1.25, 0.9])  

# out = _interp1d(np.vstack((x, y)), inter_val=0.1)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.scatter(x, y)
# ax.scatter(out[0], out[1], c='r')
# ax.legend()
# ax.set_ylabel(r"$y$", fontsize=18)
# ax.set_xlabel(r"$x$", fontsize=18)
# plt.show()