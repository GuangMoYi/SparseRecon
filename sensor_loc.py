import numpy as np
import torch
import random
import h5py
from scipy.stats import zscore
from scipy.ndimage import convolve
def cylinder_4BC_sensors():
    coords = []
    for count, i in enumerate(range(0, 112, 28)):
        if count == 0:
            continue
        coords.append([5, i])
        coords.append([187, i])

    coords = np.array(coords)

    coords = np.flip(coords, axis=1)

    return coords[[0, 1, 4, 5], 0], coords[[0, 1, 4, 5], 1]


def cylinder_8_sensors():
    coords = np.array([[76, 71], [175, 69], [138, 49],
                       [41, 56], [141, 61], [30, 41],
                       [177, 40], [80, 55]])

    coords = np.flip(coords, axis=1)

    return coords[:, 0], coords[:, 1]


def cylinder_16_sensors():
    coords = np.array([[76, 71], [175, 69], [138, 49],
                       [41, 56], [141, 61], [30, 41],
                       [177, 40], [80, 55], [60, 41], [70, 60],
                       [100, 60], [120, 51], [160, 80], [165, 50],
                       [180, 60], [30, 70]])

    coords = np.flip(coords, axis=1)

    return coords[:, 0], coords[:, 1]


def sea_n_sensors(data, n_sensors, rnd_seed):
    np.random.seed(rnd_seed)
    im = np.copy(data[0,]).squeeze()

    print('Picking up sensor locations \n')
    coords = []

    for n in range(n_sensors):
        while True:
            # new_x = np.random.randint(0, data.shape[1], 1)[0]
            # new_y = np.random.randint(0, data.shape[2], 1)[0]
            cell = 2 * 3
            new_x = np.random.randint(0, 19 + 1 + cell, 1)[0]  # (0,19)
            new_y = np.random.randint(max(0, 10 - 1 - cell), 41 + 1 + cell, 1)[0]  # (10,40)

            # Guangmo Yi modifies: 只选取海床系统周边的数据作为传感器数据
            # x_min, x_max = 10, 40
            # y_min, y_max = 0, 17
            # xd, yd = 4, 4
            # # 生成满足条件的所有可能的 (x, y) 对
            # valid_points = []
            # # 定义外部区域的边界
            # x_outer_min = x_min - xd
            # x_outer_max = x_max + xd
            # y_outer_min = y_min - yd
            # y_outer_max = y_max + yd
            # for x in range(x_outer_min, x_outer_max + xd):
            #     for y in range(y_outer_min, y_outer_max + yd):
            #         if not (x_min <= x <= x_max and y_min <= y <= y_max):
            #             valid_points.append((x, y))
            # new_x, new_y = random.choice(valid_points)

            # 使用 any() 方法来检查数组中是否有任意一个元素不等于0：
            if (im[new_x, new_y] != 0).any():
                coords.append([new_x, new_y])
                im[new_x, new_y] = 0
                break
    coords = np.array(coords)
    return coords[:, 0], coords[:, 1]


# def sea_n_sensors(data, n_sensors, rnd_seed):
#     np.random.seed(rnd_seed)
#
#     # 取出第一帧的第一个通道，并去掉单一的通道维度
#     im = np.copy(data[0,]).squeeze()
#
#     print('Picking up sensor locations \n')
#     coords = []
#
#     # 随机选择一个起点
#     while True:
#         start_x = np.random.randint(0, data.shape[1])
#         start_y = np.random.randint(0, data.shape[2])
#
#         valid = True
#         for i in range(n_sensors):
#             x = (start_x + i) % data.shape[1]
#             y = (start_y + i) % data.shape[2]
#             if im[x, y].any() == 0:
#                 valid = False
#                 break
#
#         if valid:
#             break
#
#     # 生成连续的坐标
#     for i in range(n_sensors):
#         x = (start_x + i) % data.shape[1]
#         y = (start_y + i) % data.shape[2]
#         coords.append([x, y])
#         im[x, y] = 0
#
#     coords = np.array(coords)
#     return coords[:, 0], coords[:, 1]


def sensors_3D(data, n_sensors, rnd_seed):
    np.random.seed(rnd_seed)

    im = np.copy(data[0,]).squeeze()

    print('Picking up sensor locations \n')
    coords = []

    for n in range(n_sensors):
        while True:
            new_x = np.random.randint(0, data.shape[1], 1)[0]
            new_y = np.random.randint(0, data.shape[2], 1)[0]
            new_z = np.random.randint(0, data.shape[3], 1)[0]
            if im[new_x, new_y, new_z] != 0:
                coords.append([new_x, new_y, new_z])
                im[new_x, new_y, new_z] = 0
                break
    coords = np.array(coords)
    return coords[:, 0], coords[:, 1], coords[:, 2]













