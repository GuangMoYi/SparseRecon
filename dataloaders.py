import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import torch
from scipy.stats import zscore
from scipy.ndimage import convolve
from sklearn.preprocessing import MinMaxScaler
from datasets import num1, num2
from datasets import cylinder, NOAA, pipe, plume, porous

from sensor_loc import (cylinder_16_sensors,
                        cylinder_8_sensors,
    # cylinder_4_sensors,
                        cylinder_4BC_sensors,
                        sea_n_sensors,
                        sensors_3D
                        )

import datetime
from positional import PositionalEncoder

from torch.utils.data import DataLoader, Dataset

# 注意解决这里的问题，好像没有把全局变量的值改变一样，他还是None
# START_pixels = None
def pysicalerror(pyerror, frames, pixels):
    pyerror = pyerror[frames,][:, pixels, ].float()
    # print(pixels)
    return pyerror


def GETperror(data, start_pixels_1, reshape_1, reshape_2):
    # def second_order_derivative(data, axis):
    #     """计算沿指定轴的二阶导数，忽略第一个和最后一个元素"""
    #     # 使用 numpy 的切片进行二阶差分计算
    #     second_derivative = np.zeros_like(data)
    #     dxx = 0.538473 * 0.01
    #     if axis == 1:  # 沿纵坐标（52方向）计算
    #         second_derivative[:, 1:-1, :] = (
    #                 data[:, 2:, :] - 2 * data[:, 1:-1, :] + data[:, :-2, :]
    #         )
    #         # 将第一个和最后一个元素设置为0
    #         second_derivative[:, 0, :] = 0
    #         second_derivative[:, -1, :] = 0
    #
    #     elif axis == 2:  # 沿横坐标（60方向）计算
    #         second_derivative[:, :, 1:-1] = (
    #                 data[:, :, 2:] - 2 * data[:, :, 1:-1] + data[:, :, :-2]
    #         )
    #         # 将第一个和最后一个元素设置为0
    #         second_derivative[:, :, 0] = 0
    #         second_derivative[:, :, -1] = 0
    #
    #     return second_derivative / ( dx ** 2)
    first_size, *_, last_size = data.shape
    data = data.reshape(first_size, reshape_1, reshape_2, last_size)

    # 设定各基本物理单元，注意这里有个问题是，我的数据中60代表的才是横坐标的变化（也就是axis=2）
    # 考虑一下求出的速度也是cm/s的
    data = data * num2 * 0.01
    dx = 0.538473 * 0.01
    dy = dx
    dt = 0.1   # 一开始是0.1
    data_max = 1

    # 计算分速度关于时间的导数
    dvx_dt = np.gradient(data[:, :, :, 0], dt, axis=0) / data_max
    dvy_dt = np.gradient(data[:, :, :, 1], dt, axis=0) / data_max
    # 计算水平速度Vx关于x,y轴的导数
    dvx_dx = np.gradient(data[:, :, :, 0], dx, axis=2)
    dvx_dx2 = np.gradient(data[:, :, :, 0], dx, axis=2, edge_order=2)
    dvx_dy2 = np.gradient(data[:, :, :, 0], dy, axis=1, edge_order=2)
    # dvx_dx2 = second_order_derivative(data[:, :, :, 0], axis=2)
    dvx_dy = np.gradient(data[:, :, :, 0], dx, axis=1)
    # 计算竖直速度Vy关于x,y轴的导数
    dvy_dx = np.gradient(data[:, :, :, 1], dx, axis=2)
    dvy_dy = np.gradient(data[:, :, :, 1], dy, axis=1)
    dvy_dy2 = np.gradient(data[:, :, :, 1], dy, axis=1, edge_order=2)
    dvy_dx2 = np.gradient(data[:, :, :, 1], dx, axis=2, edge_order=2)
    # 基本参数定义
    rou1 = 998
    rou2 = 1020
    mu = 0.00103
    # :rou_12 是rou1界面的，其他的是rou2界面
    rou_12 = max(0, 20 - start_pixels_1)
    error11 = rou1 * (
            dvx_dt[:, :rou_12, :] + data[:, :rou_12, :, 0] * dvx_dx[:, :rou_12, :] + data[:, :rou_12, :, 1] * dvx_dy[:,
                                                                                                              :rou_12,
                                                                                                              :]) - mu * (
                      dvx_dx2[:, :rou_12, :] + dvx_dy2[:, :rou_12, :])
    error12 = rou2 * (
            dvx_dt[:, rou_12:, :] + data[:, rou_12:, :, 0] * dvx_dx[:, rou_12:, :] + data[:, rou_12:, :, 1] * dvx_dy[:,
                                                                                                              rou_12:,
                                                                                                              :]) - mu * (
                      dvx_dx2[:, rou_12:, :] + dvx_dy2[:, rou_12:, :])
    # 计算第二个NS方程的物理误差
    error21 = rou1 * (
            dvy_dt[:, :rou_12, :] + data[:, :rou_12, :, 0] * dvy_dx[:, :rou_12, :] + data[:, :rou_12, :, 1] * dvy_dy[:,
                                                                                                              :rou_12,
                                                                                                              :]) - mu * (
                      dvy_dx2[:, :rou_12, :] + dvy_dy2[:, :rou_12, :])
    error22 = rou2 * (
            dvy_dt[:, rou_12:, :] + data[:, rou_12:, :, 0] * dvy_dx[:, rou_12:, :] + data[:, rou_12:, :, 1] * dvy_dy[:,
                                                                                                              rou_12:,
                                                                                                              :]) - mu * (
                      dvy_dx2[:, rou_12:, :] + dvy_dy2[:, rou_12:, :])

    errorns1 = np.zeros((first_size, reshape_1, reshape_2))
    errorns2 = np.zeros((first_size, reshape_1, reshape_2))
    errorns1[:, :rou_12, :] = error11
    errorns1[:, rou_12:, :] = error12
    errorns2[:, :rou_12, :] = error21
    errorns2[:, rou_12:, :] = error22
    errorut = dvx_dt
    errorvt = dvy_dt
    # erroruy2 = np.gradient(data[:, :, :, 0], dy, axis=1, edge_order=2)
    # errorvx2 = np.gradient(data[:, :, :, 1], dx, axis=2, edge_order=2)
    errorut2 = np.gradient(data[:, :, :, 0], dt, axis=0, edge_order=2)
    errorvt2 = np.gradient(data[:, :, :, 1], dt, axis=0, edge_order=2)

    # errorux2 = dvx_dx2
    # errorvy2 = dvy_dy2

    # errorux2 = second_order_derivative(data[:, :, :, 0], axis=2)
    # erroruy2 = second_order_derivative(data[:, :, :, 0], axis=1)
    # errorvx2 = second_order_derivative(data[:, :, :, 1], axis=2)
    # errorvy2 = second_order_derivative(data[:, :, :, 1], axis=1)
    errorux2 = np.gradient(data[:, :, :, 0], axis=2, edge_order=2)
    erroruy2 = np.gradient(data[:, :, :, 0], axis=1, edge_order=2)
    errorvx2 = np.gradient(data[:, :, :, 1], axis=2, edge_order=2)
    errorvy2 = np.gradient(data[:, :, :, 1], axis=1, edge_order=2)
    errorv = dvx_dx + dvy_dy
    errort = np.gradient(data, axis=0)
    errory = np.gradient(data, axis=1)
    errorx = np.gradient(data, axis=2)
    error_s = np.abs(errort) + np.abs(errory) + np.abs(errorx)
    error_smooth = error_s[:, :, :, 0] + error_s[:, :, :, 1]
    # 转化为torch类型之后再扩展一个维度，方便后续拼接
    errorns1 = torch.from_numpy(errorns1).unsqueeze(-1)
    errorns2 = torch.from_numpy(errorns2).unsqueeze(-1)
    errorut = torch.from_numpy(errorut).unsqueeze(-1)
    errorvt = torch.from_numpy(errorvt).unsqueeze(-1)
    erroruy2 = torch.from_numpy(erroruy2).unsqueeze(-1)
    errorvx2 = torch.from_numpy(errorvx2).unsqueeze(-1)
    errorux2 = torch.from_numpy(errorux2).unsqueeze(-1)
    errorvy2 = torch.from_numpy(errorvy2).unsqueeze(-1)
    errorut2 = torch.from_numpy(errorut2).unsqueeze(-1)
    errorvt2 = torch.from_numpy(errorvt2).unsqueeze(-1)
    errorv = torch.from_numpy(errorv).unsqueeze(-1)
    error_smooth = torch.from_numpy(error_smooth).unsqueeze(-1)
    # 将中间的图像坐标转化为一维度
    errorns1 = errorns1.flatten(start_dim=1, end_dim=-2)
    errorns2 = errorns2.flatten(start_dim=1, end_dim=-2)
    errorut = errorut.flatten(start_dim=1, end_dim=-2)
    errorvt = errorvt.flatten(start_dim=1, end_dim=-2)
    erroruy2 = erroruy2.flatten(start_dim=1, end_dim=-2)
    errorvx2 = errorvx2.flatten(start_dim=1, end_dim=-2)
    errorux2 = errorux2.flatten(start_dim=1, end_dim=-2)
    errorvy2 = errorvy2.flatten(start_dim=1, end_dim=-2)
    errorut2 = errorut2.flatten(start_dim=1, end_dim=-2)
    errorvt2 = errorvt2.flatten(start_dim=1, end_dim=-2)
    errorv = errorv.flatten(start_dim=1, end_dim=-2)
    error_smooth = error_smooth.flatten(start_dim=1, end_dim=-2)
    # 拼接成一个误差pyerror，并转化为float32类型
    pyerror = torch.cat([errorns1, errorns2, errorv, error_smooth, errorut, errorvt, errorux2, erroruy2, errorvx2, errorvy2, errorut2, errorvt2], axis=-1)
    return pyerror


def GETbatchrand(training_frames, batch_frames, total_pixels1, batch_pixels1, total_pixels2, batch_pixels2):
    if training_frames - batch_frames > 0:
        start_frame = torch.randint(0, training_frames - batch_frames, (1,)).item()
        frames = torch.arange(start_frame, start_frame + batch_frames)
    else:
        frames = torch.arange(batch_frames)
    if total_pixels1 - batch_pixels1 > 0 and total_pixels2 - batch_pixels2 > 0:
        start_pixels_1 = torch.randint(0, total_pixels1 - batch_pixels1, (1,)).item()
        pixels_1 = torch.arange(start_pixels_1, start_pixels_1 + batch_pixels1)
        start_pixels_2 = torch.randint(0, total_pixels2 - batch_pixels2, (1,)).item()
        pixels_2 = torch.arange(start_pixels_2, start_pixels_2 + batch_pixels2)

        pixels_1 = pixels_1.unsqueeze(1)
        pixels_2 = pixels_2.unsqueeze(0)

        pixels = pixels_1 * total_pixels2 + pixels_2
        pixels = pixels.view(-1)  # 将结果展平为一维张量
    else:
        pixels = torch.arange(batch_pixels1 * batch_pixels2)
        start_pixels_1, start_pixels_2 = 0, 0
    return frames, pixels, start_pixels_1, start_pixels_2


f = h5py.File('Data/pivdata11.mat', 'r')
# 使用 np.nan_to_num 函数将数组中的 NaN 值替换为零
shape1, shape2 = 52, 60
sst = np.nan_to_num(np.array(f['pivdata']))
num_frames, variables = sst.shape
sea = np.zeros((num_frames, shape1, shape2, 2))
kernel = np.ones((3, 3)) / 9  # 3x3 均值卷积核
for t in range(num_frames):
    reshaped_data = sst[t, :].reshape(shape1, shape2, 2, order='C')
    z_scores = zscore(reshaped_data, axis=None)
    threshold = 3
    mask = np.abs(z_scores) > threshold
    for channel in range(2):
        channel_data = reshaped_data[:, :, channel]
        local_mean = convolve(channel_data, kernel, mode='reflect')
        channel_data[mask[:, :, channel]] = local_mean[mask[:, :, channel]]
        # reshaped_data[:, :, channel] = channel_data
    # 将调整后的数据存储到结果数组中
    sea[t] = reshaped_data
sea = sea / 2
Perror = GETperror(sea, 0, shape1, shape2)

def load_data(dataset_name, num_sensors, seed=123):
    if dataset_name == 'cylinder':
        data = cylinder()

        if num_sensors == 16:
            x_sens, y_sens = cylinder_16_sensors()

        if num_sensors == 8:
            x_sens, y_sens = cylinder_8_sensors()

        if num_sensors == 4:
            # x_sens, y_sens = cylinder_4_sensors()
            pass

        if num_sensors == 4444:
            x_sens, y_sens = cylinder_4BC_sensors()

    elif dataset_name == 'sea':
        data = NOAA()
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)
    elif dataset_name == 'pipe':
        data = pipe()
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)

    elif dataset_name == 'plume':
        data = plume()
        data = data[None, :, :, :, None]
        x_sens, *y_sens = sensors_3D(data, num_sensors, seed)

    elif dataset_name == 'pore':
        data = porous()
        data = data[:, :, :, :, None]
        x_sens, *y_sens = sensors_3D(data, num_sensors, seed)

    else:
        # raise NameError('Unknown dataset')
        print(f'The dataset_name {dataset_name} was not provided\n')
        print('************WARNING************')
        print('*******************************\n')
        print('Creating a dummy dataset\n')
        print('************WARNING************')
        print('*******************************\n')
        data = np.random.rand(1000, 150, 75, 1)
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)

    print(f'Data size {data.shape}\n')
    return torch.as_tensor(data, dtype=torch.float), x_sens, y_sens


def senseiver_dataloader(data_config, num_workers=0):
    return DataLoader(senseiver_loader(data_config), batch_size=None,
                      pin_memory=True,
                      shuffle=True,
                      num_workers=num_workers
                      )


# 这个函数的具体作用则是将data_config中的各个参数利用起来，对应不同参数值进行不同的操作
class senseiver_loader(Dataset):

    def __init__(self, data_config):

        # 获取了 data_config 中的一些配置参数，如数据集名称、传感器数量和随机种子
        data_name = data_config['data_name']
        num_sensors = data_config['num_sensors']
        seed = data_config['seed']

        # 将获取的参数赋值给对象属性，并调用 load_data 函数加载相应的数据（依据data_name可以确定加载数据的内容）
        self.data_name = data_name
        self.data, x_sens, y_sens = load_data(data_name, num_sensors, seed)

        # 计算加载数据的一些信息，如总帧数、图像大小和通道数，并将其保存在 data_config 中
        total_frames, *image_size, im_ch = self.data.shape

        data_config['total_frames'] = total_frames
        data_config['image_size'] = image_size
        data_config['im_ch'] = im_ch

        # 获取训练帧数、批次帧数和批次像素数的配置信息
        self.training_frames = data_config['training_frames']
        self.batch_frames = data_config['batch_frames']
        self.batch_pixels = data_config['batch_pixels']

        # 计算总的批次数，并确保其为正数。
        num_batches = int(self.data.shape[1:].numel() * self.training_frames / (
                self.batch_frames * self.batch_pixels))

        assert num_batches > 0

        # 输出数据集的批次数，并将其保存在 data_config 和对象属性中
        print(f'{num_batches} Batches of data per epoch\n')
        data_config['num_batches'] = num_batches
        self.num_batches = num_batches

        # 根据是否连续训练的配置信息，初始化训练数据的索引
        if data_config['consecutive_train']:
            self.train_ind = torch.arange(0, self.training_frames)
        else:
            if seed:
                torch.manual_seed(seed)
            self.train_ind = torch.randperm(self.data.shape[0])[:self.training_frames]

        # 如果批次帧数大于训练帧数，输出警告，并将批次帧数设置为训练帧数。
        if self.batch_frames > self.training_frames:
            print('Warning: batch_frames bigger than num training samples')
            self.batch_frames = self.training_frames

        # 创建传感器坐标，得到非零元素的索引。
        sensors = np.zeros(self.data.shape[1:-1])

        if len(sensors.shape) == 2:
            sensors[x_sens, y_sens] = 1
        elif len(sensors.shape) == 3:  # 3D images
            sensors[x_sens, y_sens[0], y_sens[1]] = 1

        self.sensors, *_ = np.where(sensors.flatten() == 1)  # 矩阵扁平化处理，变成一维数据，方便判断矩阵位置（==1）

        # 创建位置编码器（PositionalEncoder）
        self.pos_encodings = PositionalEncoder(self.data.shape[1:], data_config['space_bands'])

        # 对选中的传感器进行位置编码，第三行代码复制了位置编码batch_frames次，表示每张图片的位置编码是一样的
        self.indexed_sensors = self.data.flatten(start_dim=1, end_dim=-2)[:, self.sensors, ]
        self.sensor_positions = self.pos_encodings[self.sensors,]
        self.sensor_positions = self.sensor_positions[None,].repeat_interleave(
            self.batch_frames, axis=0)

        # 获取非零像素的索引（并且是变为一个列向量之后的索引）
        self.pix_avail = self.data.flatten(start_dim=1, end_dim=-2)[0, :, 0].nonzero()[:, 0]

        # 如果存在随机种子，重置 PyTorch 的随机种子
        if seed:
            torch.manual_seed(datetime.datetime.now().microsecond)

    # 返回数据集的总批次数
    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # 随机选择 self.batch_frames 个帧的索引    (每批次取这么多帧)
        # 随机选择 self.batch_pixels 个像素的索引 （每批次取那么多像素）
        # frames = self.train_ind[ torch.randperm( self.training_frames) ][:self.batch_frames]
        # pixels = self.pix_avail[ torch.randperm(*self.pix_avail.shape) ][:self.batch_pixels]

        # # Guangmo Yi modifies:
        batch_pixels1, batch_pixels2 = 32, 32
        frames, pixels, self.start_pixels1, start_pixels2 = GETbatchrand(self.training_frames, self.batch_frames, shape1, batch_pixels1, shape2, batch_pixels2)
        give_pixels = self.start_pixels1 * torch.ones(self.batch_frames, batch_pixels1 * batch_pixels2, 1)
        give_pixels[0, 1, 0] = start_pixels2

        #  self.indexed_sensors 中获取选定帧 frames 的传感器值
        sensor_values = self.indexed_sensors[frames,]

        # 将传感器值与传感器位置编码 self.sensor_positions 沿着最后一个轴（axis=-1）进行拼接
        sensor_values = torch.cat([sensor_values, self.sensor_positions], axis=-1)

        # 如果 self.data_name 是 'pipe'，则在传感器值中随机选择 rnd_sensor_num 个传感器的索引，并仅保留这些传感器的数据
        if self.data_name == 'pipe':
            rnd_sensor_num = (40 + 300 * torch.abs(torch.randn(1))).type(torch.int)
            rnd_sensor_ind = torch.randperm(6144)[:rnd_sensor_num]
            sensor_values = sensor_values[:, rnd_sensor_ind, :]

        # 从位置编码 self.pos_encodings 中获取选定像素 pixels 的坐标编码，扩展为形状为 (self.batch_frames, ...)
        coords = self.pos_encodings[pixels,][None,]
        coords = coords.repeat_interleave(self.batch_frames, axis=0)
        # 从原始数据 self.data 中获取选定帧 frames 和像素 pixels 的场值
        field_values = self.data.flatten(start_dim=1, end_dim=-2)[frames,][:, pixels, ]

        # Guangmo Yi modifies: 为了增加物理限制
        pyerror = pysicalerror(Perror, frames, pixels)
        field_values = torch.cat([field_values, pyerror], axis=-1)
        # 这里的第五个维度就是要传过去的随机开始的图像pixels1
        field_values = torch.cat([field_values, give_pixels], axis=-1)

        # sensor_values --> torch.Size([64, 8, 129])：一批次下的帧的选定的传感器的值和位置的组合
        # coords --> torch.Size([64, 2048, 128]): 一批次下的帧的一批次的像素点的位置编码
        # field_values --> torch.Size([64, 2048, 1]): 一批次下的帧的一批次的像素点的值

        return sensor_values, coords, field_values



