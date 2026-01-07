import numpy as np
import pickle as pk
from dataloaders import GETperror
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from tqdm import tqdm as bar
import h5py
from model import Encoder, Decoder
from scipy.stats import zscore
from scipy.ndimage import convolve
from scipy.stats import wasserstein_distance


f = h5py.File('Data/pivdata11.mat', 'r')
sst = np.nan_to_num(np.array(f['pivdata']))
num_frames, variables = sst.shape
shape1, shape2 = 52, 60
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
pre = torch.from_numpy(sea).float()


# 定义了名为 Senseiver 的类，继承自 PyTorch Lightning 的 pl.LightningModule
class Senseiver(pl.LightningModule):
    def __init__(self, **kwargs):
        # 使用 self.save_hyperparameters() 保存所有传递给模型的超参数（也就是data_config, encoder_config, decoder_config中的参数）
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        # 计算位置编码的通道数，self.hparams.space_bands一般是输入数据的空间维度数，image_size是输入数据的形状数目，二维张量即=2
        pos_encoder_ch = self.hparams.space_bands * len(self.hparams.image_size) * 2
        print("Pos_encoder_ch: ", pos_encoder_ch)
        print("Im_ch: ", self.hparams.im_ch)
        # 实例化了名为 Encoder 的对象，并传递了一系列参数作为构造函数的参数。
        self.encoder = Encoder(
            input_ch=self.hparams.im_ch + pos_encoder_ch,
            preproc_ch=self.hparams.enc_preproc_ch,
            num_latents=self.hparams.num_latents,
            num_latent_channels=self.hparams.enc_num_latent_channels,
            num_layers=self.hparams.num_layers,
            num_cross_attention_heads=self.hparams.num_cross_attention_heads,
            num_self_attention_heads=self.hparams.enc_num_self_attention_heads,
            num_self_attention_layers_per_block=self.hparams.num_self_attention_layers_per_block,
            dropout=self.hparams.dropout,
        )

        # 这行代码类似于前面的 Encoder，实例化了一个名为 Decoder 的对象，同样传递了一系列参数
        self.decoder_1 = Decoder(
            ff_channels=pos_encoder_ch,
            preproc_ch=self.hparams.dec_preproc_ch,  # latent bottleneck
            num_latent_channels=self.hparams.dec_num_latent_channels,  # hyperparam
            latent_size=self.hparams.latent_size,  # collapse from n_sensors to 1
            num_output_channels=self.hparams.im_ch,
            num_cross_attention_heads=self.hparams.dec_num_cross_attention_heads,
            dropout=self.hparams.dropout,
        )

        # 使用 filter 函数过滤出模型中需要梯度更新的参数。p.requires_grad 用于检查参数是否需要梯度，self.parameters()返回模型的所有参数
        # 使用列表推导式计算每个参数张量的元素数量（通过 np.prod(p.size()) 得到），然后使用 sum 函数将所有参数的元素数量相加，得到总的参数数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'\nThe model has {self.num_params} params \n')

        # Guangmo Yi modifies:
        loss_value = []
        self.loss_value = loss_value
        error_weight = 0
        self.error_weight1 = error_weight
        self.error_weight2 = error_weight
        self.error_weight3 = error_weight
        self.error_weight4 = error_weight
        self.error_weight5 = error_weight
        self.error_weight6 = error_weight
        self.error_weight7 = error_weight
        epoch_use = 0
        self.epoch_use = epoch_use

    # 整个模型的运转过程
    def forward(self, sensor_values, query_coords):
        out = self.encoder(sensor_values)
        return self.decoder_1(out, query_coords)

    # 用于定义模型在训练过程中的一步操作，返回的是训练误差
    # batch: 训练时从数据加载器中获取的一个批次数据。
    # batch_idx: 当前批次的索引
    def training_step(self, batch, batch_idx):
        # 从批次数据中解包得到传感器值 (sensor_values)、查询坐标 (coords) 和目标场值 (field_values)
        sensor_values, coords, field_values = batch  # Every epoch have 36 batch_id: 0-35
        pred_values = self(sensor_values, coords)
        # Guangmo Yi modifies:
        # # # 只取物理限制
        pyerror = field_values[:, :, 2:-1]
        # # 获得数据中随机开始的图像pixels1点数，并转换为整数类型
        get_pixels1 = field_values[0, 0, -1].int()
        get_pixels2 = field_values[0, 1, -1].int()
        field_values = field_values[:, :, :2]

        # loss = F.mse_loss(pred_values, field_values, reduction='sum')

        def quantile_loss(pred, target, quantile):
            diff = target - pred
            return torch.sum(torch.max(quantile * diff, (quantile - 1) * diff))

        loss = quantile_loss(pred_values, field_values, quantile=0.5)
        # print(loss)





        def error_smooth(error, start_pixels_1, start_pixels_2):
            if 18 - start_pixels_1 >= 0 and 40 - start_pixels_2 >= 0:
                error[[max(17 - start_pixels_1, 0), 18 - start_pixels_1],
                max(9 - start_pixels_2, 0):40 - start_pixels_2, :] = 0
            if 1 - start_pixels_1 >= 0 and 37 - start_pixels_2 >= 0:
                error[[0, 1 - start_pixels_1], max(11 - start_pixels_2, 0):37 - start_pixels_2, :] = 0
            if 14 - start_pixels_1 >= 0 and 12 - start_pixels_2 >= 0:
                error[[max(13 - start_pixels_1, 0), 14 - start_pixels_1],
                max(9 - start_pixels_2, 0):12 - start_pixels_2, :] = 0
            if 14 - start_pixels_1 >= 0 and 40 - start_pixels_2 >= 0:
                error[[max(13 - start_pixels_1, 0), 14 - start_pixels_1],
                max(36 - start_pixels_2, 0):40 - start_pixels_2, :] = 0
            if 18 - start_pixels_1 >= 0 and 10 - start_pixels_2 >= 0:
                error[max(13 - start_pixels_1, 0):18 - start_pixels_1,
                [max(9 - start_pixels_2, 0), 10 - start_pixels_2], :] = 0
            if 18 - start_pixels_1 >= 0 and 40 - start_pixels_2 >= 0:
                error[max(13 - start_pixels_1, 0):18 - start_pixels_1,
                [max(39 - start_pixels_2, 0), 40 - start_pixels_2], :] = 0
            if 14 - start_pixels_1 >= 0 and 12 - start_pixels_2 >= 0:
                error[:14 - start_pixels_1, [max(11 - start_pixels_2, 0), 12 - start_pixels_2], :] = 0
            if 14 - start_pixels_1 >= 0 and 37 - start_pixels_2 >= 0:
                error[:14 - start_pixels_1, [max(36 - start_pixels_2, 0), 37 - start_pixels_2], :] = 0
            return error

        # Guangmo Yi modifies:
        # 先讨论前面没有权重的情况
        opt_change = -100
        if self.current_epoch <= opt_change:
            pred_values_cpu = pred_values
            # 将张量复制后取消其计算图并转化为numpy类型
            pred_values_cpu = pred_values_cpu.detach().cpu().numpy()
            batch_pixels1, batch_pixels2 = 32, 32
            pyerror_now = GETperror(pred_values_cpu, get_pixels1, batch_pixels1, batch_pixels2)
            # 将张量转化到GPU中运算
            pyerror_now = pyerror_now.to("cuda")
            pyerror1 = pyerror[:, :, :2]
            pyerror2 = pyerror[:, :, 2:3]
            pyerror3 = pyerror[:, :, 3:4]
            pyerror4 = pyerror[:, :, 4:6]
            pyerror5 = pyerror[:, :, 6:8]
            pyerror6 = pyerror[:, :, 8:10]
            pyerror7 = pyerror[:, :, 10:]
            pyerror_now1 = pyerror_now[:, :, :2]
            pyerror_now2 = pyerror_now[:, :, 2:3]
            pyerror_now3 = pyerror_now[:, :, 3:4]
            pyerror_now4 = pyerror_now[:, :, 4:6]
            pyerror_now5 = pyerror_now[:, :, 6:8]
            pyerror_now6 = pyerror_now[:, :, 8:10]
            pyerror_now7 = pyerror_now[:, :, 10:]
            pyerror_now3 = error_smooth(pyerror_now3, get_pixels1, get_pixels2)


            extra_loss1 = quantile_loss(pyerror_now1[(pyerror1 <= 50)], pyerror1[(pyerror1 <= 50)], quantile=0.5)
            extra_loss2 = quantile_loss(pyerror_now2, pyerror2, quantile=0.5)
            extra_loss3 = torch.sum(pyerror_now3)
            extra_loss4 = quantile_loss(pyerror_now4, pyerror4, quantile=0.5)
            extra_loss5 = quantile_loss(pyerror_now5, pyerror5, quantile=0.5)
            extra_loss6 = quantile_loss(pyerror_now6, pyerror6, quantile=0.5)
            extra_loss7 = quantile_loss(pyerror_now7, pyerror7, quantile=0.5)
            self.error_weight1 = 1e-4                 # 1e-2; 0.0005(1e-4)-->1e-4:0.47
            self.error_weight2 = 0                    # 2(1)-->1e-2, 0.62[4]
            self.error_weight3 = 0                    # 1000, 0.60
            self.error_weight4 = 0                    # 100:0.46, 1000:0.52  10:0.52
            self.error_weight5 = 0                    # 1e-1(or 1) 1:0.56  1e-1:0.48 1e-2:0.52
            self.error_weight6 = 0                    # 1e-1-->1:0.57 1e-1:0.53 1e-2:0.46
            self.error_weight7 = 0                    # 1, 0.63
            # 0.1*1e-4  0.5*5e-1  0  0.1*1000  0   0  0.1*100 这是0.78

            # print("extra_loss1:", extra_loss1)
            # print("loss:", loss)

            loss = (loss + self.error_weight1 * extra_loss1 + self.error_weight2 * extra_loss2 +
                    self.error_weight3 * extra_loss3 + self.error_weight4 * extra_loss4 +
                    self.error_weight5 * extra_loss5 + self.error_weight6 * extra_loss6 +
                    self.error_weight7 * extra_loss7)
       # [errorns1, errorns2, errorv, error_smooth, errorut, errorvt, errorux2, erroruy2, errorvx2, errorvy2,
       #       errorut2, errorvt2]

        # 使用 PyTorch Lightning 提供的日志记录功能，记录训练损失。
        # loss/field_values.numel(): 归一化的损失，除以目标场值的元素数量。
        # batch_size=1: 指定批次大小。
        self.log("train_loss", loss/field_values.numel(),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=1)
        return loss

    def configure_optimizers(self):
        # 初始优化器，使用 Adam 优化器，并从超参数中获取初始学习率
        optimizer = torch.optim.Adam(self.parameters(), lr=5 * self.hparams.lr)

        # CosineAnnealingLR 调度器，初始学习率为 self.hparams.lr，T_max 是训练周期数，
        # eta_min 是学习率的最小值，通常设置为一个很小的数，如 1e-6
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=300,
                                                               eta_min=1e-6)

        # 返回优化器和调度器
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # 每个epoch调用一次调度器
                'frequency': 1  # 每个epoch更新一次学习率
            }
        }

    # # 配置一个 Adam 优化器，用于训练模型权重
    # def configure_optimizers(self):
    #     # Guangmo Yℹ modifies： 为了修改优化器
    #     # opt_change = 300
    #     # if self.current_epoch < opt_change:
    #     # lr_min, lr_max, epoch_max = 1e-5, 10 * self.hparams.lr, 1000
    #     # lr_now = lr_min + 1 / 2 * (lr_max - lr_min) * (1 + np.cos(self.current_epoch * np.pi / epoch_max))
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    #     # else:
    #     #     optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
    #     return optimizer


    # 该方法用于测试模型
    def test(self, dataloader, num_pix=1024, split_time=0):

        # 获取数据集的形状信息：包括数据集的数量 (im_num)、图像大小 (im_size) 和通道数 (im_ch)
        im_num, *im_size, im_ch = dataloader.dataset.data.shape

        im_pix = np.prod(im_size)  # 计算该数组中所有m * n个元素的乘积，这里是将其图片信息计算为1维的个数，即52，60-->3120
        pixels = np.arange(0, im_pix, num_pix)  # 创建一个一维数组 pixels，其中包含从0开始、以 num_pix 为步长、不超过 im_pix 的所有像素的索引
        output_im = torch.zeros(im_num, im_pix, im_ch)
        sca_factor = 0.3
        bias = 1
        # 根据数据集的数量 (im_num) 决定时间步的划分。如果数据集只有一个图像，设置 times 为 [0, 1]，
        # 否则使用 split_time 参数在时间维度上均匀划分
        if im_num == 1:
            times = [0, ]
        else:
            times = np.linspace(0, im_num, split_time, dtype=int)

        # data 从数据加载器中获取图像数据 (im)、传感器信息 (sensors) 和位置编码信息 (pos_encodings)
        im = dataloader.dataset.data
        sensors = dataloader.dataset.sensors
        pos_encodings = dataloader.dataset.pos_encodings

        # 迭代遍历时间步 (t_start) ，对每一个时刻，遍历图像像素 (pix)：代码会从 times 列表的第二个元素开始遍历
        # 从位置编码中获取坐标信息 (coords)，根据时间步的间隔进行复制
        # 从图像数据中获取相应时间步内的传感器值 (sensor_values)。
        # 从位置编码中获取传感器位置信息 (sensor_positions)，根据传感器数量进行复制。
        # 将传感器值和位置信息拼接在一起，传递给模型 (self) 进行测试，得到输出 (out)。
        # 将输出填充到 output_im 中
        t = 0
        for t_start in bar(times[1:]):  # 使用 tqdm 库中的 bar 函数实现的，提供可视化的进度条，显示循环的进行程度
            dt = t_start - t
            for pix in bar(pixels):
                # 选择从 pix 到 pix+num_pix 行的位置编码子集，None 用于在第一个维度上添加一个新的维度，形成一个(num_pix, 编码维度) 的二维张量
                # 沿着第一个轴（axis=0）重复张量 coords，重复次数为 dt,结果是一个形状为 (dt * num_pix, 编码维度) 的张量。
                coords = pos_encodings[pix:pix + num_pix, ][None,]
                coords = coords.repeat_interleave(dt, axis=0)

                # 从第二个维度（索引为1的维度）开始扁平化，而 end_dim=-2 表示扁平化到倒数第二个维度为止,形成(第一维度：图片数量,扁平数据,最后一维度)的二维张量
                sensor_values = im.flatten(start_dim=1, end_dim=-2)[t:t_start, sensors]

                sensor_positions = pos_encodings[sensors,][None,]
                sensor_positions = sensor_positions.repeat_interleave(sensor_values.shape[0], axis=0)

                sensor_values = torch.cat([sensor_values, sensor_positions], axis=-1)
                # 代入传感器值，完成整个模型过程，得到out
                out = self(sensor_values, coords)

                output_im[t:t_start, pix:pix + num_pix] = out
            t += dt

        # 将输出的形状调整为图像形状，并将数据集中值为零的位置数值保留。
        output_im = output_im.reshape(-1, *im_size, im_ch)
        # # 假设 output_im 在 GPU 上
        data_device = output_im.device
        dataloader.dataset.data = dataloader.dataset.data.to(data_device)
        x1 = 9
        x2 = 12
        x3 = 36
        x4 = 39
        # y1 = 0
        y2 = 13
        y3 = 17
        # 返回测试结果，即模型在给定数据集上的输出图像
        output_im[:, y2:y3, x1:x4, :] = 0
        output_im[:, :y2, x2:x3, :] = 0
        output_im[dataloader.dataset.data == 0] *= bias
        adjust = torch.norm(dataloader.dataset.data[dataloader.dataset.data != 0]) / torch.norm(
            output_im[dataloader.dataset.data != 0])
        output_im[dataloader.dataset.data == 0] *= adjust
        output_im = (1 - sca_factor) * output_im + sca_factor * pre
        # output_im[dataloader.dataset.data != 0] = 0.5 * output_im[dataloader.dataset.data != 0] + 0.5 * dataloader.dataset.data[dataloader.dataset.data != 0]
        # output_im[dataloader.dataset.data != 0] = dataloader.dataset.data[dataloader.dataset.data != 0]

        return output_im

    def histogram(self, path):
        import pickle

        results = dict()
        # 使用 torch.no_grad() 上下文管理器，关闭梯度计算，因为这里只是测试而不是训练
        with torch.no_grad():

            # 设置 im_num 为 500，然后对 self.im 进行切片，取前500个
            self.im_num = 509  # 之前是500
            self.im = self.im[:self.im_num]

            # pixels 是一个包含从0到self.im_pix的数组。coords 是模型的位置编码
            pixels = np.arange(0, self.im_pix)
            coords = self.pos_encoder.position_encoding[:, ][None,]

            # 在这个双重循环中，对于每个随机种子和传感器数量的组合，都进行以下操作：
            for seed in bar([123, 1234, 12345, 9876, 98765, 666, 777, 11111]):
                results[str(seed)] = {}
                for num_of_sensors in [25, 50, 100, 150, 200, 250, 500, 750]:

                    # 使用torch.manual_seed设置随机种子，然后使用torch.randperm生成6144个随机索引，
                    # 再取其中的前num_of_sensors个，作为传感器的索引
                    torch.manual_seed(seed)
                    rnd_sensor_ind = torch.randperm(6144)[:num_of_sensors]

                    # 创建一个用于存储模型预测的初始零张量
                    pred = torch.zeros(self.im_num, self.im_pix, 1)

                    # 获取模型中随机选取的传感器位置的位置编码，self.sensors[rnd_sensor_ind] 选择了前面确定的那个传感器索引子集
                    sensor_positions = self.pos_encoder.position_encoding[self.sensors[rnd_sensor_ind],][None,]

                    # 在这个循环中，对于每张图片和像素位置，进行以下操作：
                    # 从输入数据中提取传感器值，获取传感器的位置编码，将它们合并，并用模型进行预测，将结果存储在pred中
                    for pix in range(self.im_num):
                        sensor_values = self.im.flatten(start_dim=1, end_dim=2)[pix:pix + 1,
                                        self.sensors[rnd_sensor_ind]]
                        sensor_values = torch.cat([sensor_values, sensor_positions], axis=-1)
                        pred[pix, :] = self(sensor_values, coords)

                    # 将 pred 的形状重新调整为 (self.im_num, *self.im_dims, self.im_ch)，这样它就可以被比较和分析
                    # 计算预测值与真实值之间的误差，并将结果存储在results字典中
                    pred = pred.reshape(-1, *self.im_dims, self.im_ch)
                    e = (self.im.cpu() - pred).norm(p=2, dim=(1, 2)) / (self.im.cpu()).norm(p=2, dim=(1, 2))
                    results[str(seed)][str(num_of_sensors)] = e.mean()
                print(results)

        #  使用pickle将results保存到二进制文件中，以便稍后进行分析
        with open(f'{path}/errors.pk', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)










