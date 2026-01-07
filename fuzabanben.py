import numpy as np
import h5py
from glob import glob as gb
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from s_parser import parse_args
from dataloaders import senseiver_dataloader
from network_light import Senseiver
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from plot import plot_cs
# from plot import plot_all_ts
import time
from scipy.ndimage import convolve
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.stats import zscore
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
import torch.nn.init as init
# Train-->Test:  train: num_iterations=100      s_parser: ##           datasets: pivdataTestdata1

# 记录开始时间
start_time = time.time()
# 初始化一个列表，用于存储每次迭代的均方差
# Guangmo Yi modifies:
f = h5py.File('Data/pivdata11.mat', 'r')
sst = np.nan_to_num(np.array(f['pivdata']))
num_frames, variables = sst.shape
shape1, shape2 = 52, 60
sea = np.zeros((num_frames, shape1, shape2, 2))
# 定义卷积核，用于计算周围点的均值
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
true = sea

f = h5py.File('Data/5093122pivdata11.mat', 'r')
sst = np.nan_to_num(np.array(f['pivdata']))
num_frames, variables = sst.shape
sea = np.zeros((num_frames, shape1, shape2, 2))
for t in range(num_frames):
    reshaped_data = sst[t, :].reshape(shape1, shape2, 2, order='C')
    z_scores = zscore(reshaped_data, axis=None)
    threshold = 3
    mask = np.abs(z_scores) > threshold
    for channel in range(2):
        channel_data = reshaped_data[:, :, channel]
        local_mean = convolve(channel_data, kernel, mode='reflect')
        channel_data[mask[:, :, channel]] = local_mean[mask[:, :, channel]]
        # reshaped_data[:, :, channel] = channe_data
        # 将调整后的数据存储到结果数组中
    sea[t] = reshaped_data
sea = sea / 2
YS_figure = torch.tensor(sea).detach()
mse_values = []
# mae_values = []
# psd_values = []
# gof_values = []
# pde_values = []

# 设置循环次数
num_iterations = 1
# 2-4: e-5
# v70-72 0.441 基本
# v78 0.43  MLP+Residual
# v92 0.440, space 8 training 1100
# V84 0.70 --> 出现过0.66
# V85,89 0.70 , enc_num_latent_channels = 64 0.33
# V90， 无物理限制
# V91， 无RNN  0.1
# V92， 无CNN  0.25
# V93， (1, 0, 0) 0.1
# V94， (0, 1, 0) 0.2
# V95， (0, 0, 1) 0.15
# 1 0.59
for num in range(num_iterations):
    def initialize_weights(module):
        try:
            if isinstance(module, nn.Conv1d):
                if module.weight is not None and module.weight.dim() >= 2:
                    init.kaiming_normal_(module.weight, nonlinearity='relu')  # 使用 He 初始化
                if module.bias is not None:
                    init.constant_(module.bias, 0.01)  # 偏置初始化为小的正数

            elif isinstance(module, nn.Linear):
                if module.weight is not None and module.weight.dim() >= 2:
                    init.kaiming_normal_(module.weight, nonlinearity='relu')  # 使用 He 初始化
                if module.bias is not None:
                    init.constant_(module.bias, 0.01)  # 偏置初始化为小的正数

            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    init.ones_(module.weight)  # 权重初始化为 1
                if module.bias is not None:
                    init.constant_(module.bias, 0.01)  # 偏置初始化为小的正数

            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    init.ones_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0.01)

            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    init.ones_(module.weight)  # 对于LayerNorm，初始化为 1
                if module.bias is not None:
                    init.constant_(module.bias, 0.01)  # 偏置初始化为小的正数

            elif isinstance(module, nn.LSTM):
                # 初始化 LSTM 权重
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        init.kaiming_normal_(param, nonlinearity='relu')  # 输入门权重使用 He 初始化
                    elif 'weight_hh' in name:
                        init.kaiming_normal_(param, nonlinearity='relu')  # 隐藏门权重使用 He 初始化
                    elif 'bias' in name:
                        init.constant_(param, 0)  # 偏置初始化为 0

        except Exception as e:
            print(f"初始化模块时出错: {module}")
            print(f"权重形状: {getattr(module.weight, 'shape', None)}")
            print(f"偏置形状: {getattr(module.bias, 'shape', None)}")
            raise e


    data_config, encoder_config, decoder_config = parse_args()

    # 将data_config中的参数利用上，通过senseiver_dataloader，得到对应的数据
    dataloader = senseiver_dataloader(data_config, num_workers=0)

    checkpoint_path = '.\lightning_logs/version_1/checkpoints/train-epoch=599.ckpt'  # 修改为实际文件路径

    model = Senseiver(
        **encoder_config,
        **decoder_config,
        **data_config
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    # model.apply(initialize_weights)  # 遍历所有子模块并应用初始化

    # 如果指定了加载模型的模型编号，则加载模型
    if encoder_config['load_model_num']:
        model_num = encoder_config['load_model_num']
        model_num_new = model_num + num
        print(f'Loading {model_num_new} ...')

        model_loc = gb(f"lightning_logs/version_{model_num_new}/checkpoints/*.ckpt")[0]

        # Guangmo Yi modifies: 问题在于不能在实例model上调用 load_from_checkpoint，而它实际上应该在类Senseiver上调用。
        model = Senseiver.load_from_checkpoint(model_loc,
                                               **encoder_config,
                                               **decoder_config,
                                               **data_config)
    else:
        model_loc = None

    # 如果训练阶段，设置回调函数（callbacks：cbs）并使用Trainer类进行模型训练
    if not data_config['test']:
        # ModelCheckpoint: 用于在训练过程中保存模型的权重。
        # monitor="train_loss" 指定监测的指标为训练过程中的损失函数
        # filename="train-{epoch:02d}" 设置保存的模型文件名，其中 {epoch:02d} 表示保存时使用两位十进制数
        # every_n_epochs=10 表示每隔10个训练周期保存一次模型
        # save_on_train_epoch_end=True 表示在每个训练周期结束时都保存一次模型

        # EarlyStopping: 用于在训练过程中监视train_loss，并在达到某个条件时停止训练，以避免过拟合。
        # check_finite=False 设置为 False 表示在监测指标中允许非有限值，通常用于处理梯度爆炸等情况
        # patience=100 表示如果在连续100个训练周期内监测指标没有改善，则停止训练[注意，现在改成了50，原来是100]
        cbs = [ModelCheckpoint(monitor="train_loss", filename="train-{epoch:02d}",
                               every_n_epochs=10, save_on_train_epoch_end=True),
               EarlyStopping(monitor="train_loss", check_finite=False, patience=100)]     # 这里删掉了patience=100

        # trainer = Trainer(max_epochs=-1,
        #                   callbacks=cbs,
        #                   gpus=data_config['gpu_device'],
        #                   accumulate_grad_batches=data_config['accum_grads'],
        #                   log_every_n_steps=data_config['num_batches'],
        #                   )

        # Guangmo Yi modifies:
        # max_epochs=-1：表示训练的最大周期数。设置为 -1 表示训练将在达到提前停止条件或手动中断前一直进行，不受周期数的限制，这里改为了1000
        # accelerator='auto'：表示使用自动选择的加速器。加速器用于加速模型训练
        # accumulate_grad_batches=data_config['accum_grads']：指定累积梯度的批次数。累积梯度是一种训练技巧，允许在更新模型参数之前累积多个小批次的梯度，有助于稳定训练过程。data_config['accum_grads'] 从配置文件中获取该值
        trainer = Trainer(max_epochs=-1,
                          callbacks=cbs,
                          accelerator='auto',
                          accumulate_grad_batches=data_config['accum_grads'],
                          log_every_n_steps=data_config['num_batches'],
                          )
        # 这个函数调用的是一个训练器（trainer）的 fit 方法
        # 使用给定的数据对模型进行训练。这可能包括多个训练轮次，每个轮次使用 dataloader 加载数据，
        # 并通过反向传播和优化算法来更新模型的参数。检查点路径 ckpt_path 可能用于定期保存模型的状态（测试时候用的）。
        trainer.fit(model, dataloader,
                    ckpt_path=model_loc
                    )
    # 如果是测试阶段，将模型移动到GPU设备上（如果指定了GPU），然后使用模型测试数据
    else:
        if data_config['gpu_device']:
            device = data_config['gpu_device'][0]
            model = model.to(f"cuda:{device}")

            model = model.to(f"cuda:{data_config['gpu_device'][0]}")
            dataloader.dataset.data = torch.as_tensor(
                dataloader.dataset.data).to(f"cuda:{device}")
            dataloader.dataset.sensors = torch.as_tensor(
                dataloader.dataset.sensors).to(f"cuda:{device}")
            dataloader.dataset.pos_encodings = torch.as_tensor(
                dataloader.dataset.pos_encodings).to(f"cuda:{device}")
        # model_loc.split('checkpoints') 使用字符串的 split 方法，以 "checkpoints" 为分隔符将 model_loc 字符串分割成一个列表。
        # [0] 选择了分割后列表的第一个元素，即 "checkpoints" 之前的部分。
        path = model_loc.split('checkpoints')[0]

        with torch.no_grad():
            output_im = model.test(dataloader, num_pix=1024, split_time=10)   # 从2048改成了 num_pix = 1024
        torch.save(output_im, f'{path}/res.torch')


        output_im = output_im.clone().detach()
        # zero_count = (output_im == 0).sum().item()
        # print(f"Number of zeros in output_im: {zero_count}")
        true       = torch.tensor(true).detach()
        # 初始化一个列表用于存储每对张量的均方差
        mse_list = []
        mae_list = []
        psd_list = []
        gof_list = []
        pde_list = []
        output_set = output_im
        true_set = true

        # 遍历测试集和训练集中的每一对张量
        for test_tensor, true_tensor in zip(output_set, true_set):
            # 计算均方差
            # mse = torch.nn.functional.mse_loss(test_tensor, true_tensor)
            # 计算mse误差值
            mse = torch.norm(true_tensor-test_tensor)/torch.norm(true_tensor)
                   # /true_tensor.numel()

            # # 计算 MAE
            # mae = torch.mean(torch.abs(true_tensor - test_tensor))
            # # 计算 PSD（Profile Standard Deviation）
            # psd = torch.std(true_tensor - test_tensor)
            # # 计算 Goodness of Fit (Gof)  越大越好
            # true_mean = torch.mean(true_tensor)
            # ss_total = torch.sum((true_tensor - true_mean) ** 2)
            # ss_residual = torch.sum((true_tensor - test_tensor) ** 2)
            # gof = 1 - (ss_residual / ss_total)
            # 计算 Peak Depth Error (PDE)
            # max_true = torch.max(torch.abs(true_tensor))
            # max_test = torch.max(torch.abs(test_tensor))
            # pde = torch.mean(torch.abs(max_true - max_test))
            # 检查是否是有限值（排除无穷大和NaN）
            if torch.isfinite(mse):
                mse_list.append(mse.item())
            # if torch.isfinite(mae):
            #     mae_list.append(mae.item())
            # if torch.isfinite(psd):
            #     psd_list.append(psd.item())
            # if torch.isfinite(gof):
            #     gof_list.append(gof.item())
            # if torch.isfinite(pde):
            #     pde_list.append(pde.item())
        # 计算均方差的平均值
        if len(mse_list) == 0:
            average_mse = 1  # 或者给一个默认值
        else:
            average_mse = sum(mse_list) / len(mse_list)
        # average_mae = sum(mae_list) / len(mae_list)
        # average_psd = sum(psd_list) / len(psd_list)
        # average_gof = sum(gof_list) / len(gof_list)
        # average_pde = sum(pde_list) / len(pde_list)
        # 将均方差添加到列表中
        mse_values.append(average_mse)
        # mae_values.append(average_mae)
        # psd_values.append(average_psd)
        # gof_values.append(average_gof)
        # pde_values.append(average_pde)

        # ######## 展示预测的最后一组图片的预测效果
        # if num == num_iterations - 1:
        #     for j in range(true_set.shape[0]):
        #         image1 = true_set[j, :, :, 0].numpy()
        #         image2 = output_set[j, :, :, 0].numpy()
        #         error_set = true_set[j, :, :, 0] - output_set[j, :, :, 0]
        #         image3 = YS_figure[j, :, :, 0].numpy()
        #         image4 = error_set.numpy()
        #
        #         cmap = plt.cm.seismic
        #
        #         # 为前三个图使用的比例尺
        #         norm1 = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        #         # 为最后一个图使用的比例尺
        #         norm2 = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)  # 根据需要调整
        #
        #         # 创建四个子图
        #         fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        #
        #         # 在第一个子图中绘制图像
        #         axs[0].imshow(image1, cmap=cmap, norm=norm1)
        #         axs[0].axis('off')  # 关闭坐标轴
        #         axs[0].set_title('PIV', fontdict={'family': 'Times New Roman', 'fontsize': 16})
        #
        #         # 在第二个子图中绘制图像
        #         axs[1].imshow(image2, cmap=cmap, norm=norm1)
        #         axs[1].axis('off')  # 关闭坐标轴
        #         axs[1].set_title('Reconstruction', fontdict={'family': 'Times New Roman', 'fontsize': 16})
        #
        #         # 在第三个子图中绘制图像
        #         axs[2].imshow(image3, cmap=cmap, norm=norm1)
        #         axs[2].axis('off')  # 关闭坐标轴
        #         axs[2].set_title('Test', fontdict={'family': 'Times New Roman', 'fontsize': 16})
        #
        #         # 在第四个子图中绘制图像
        #         axs[3].imshow(image4, cmap=cmap, norm=norm2)
        #         axs[3].axis('off')  # 关闭坐标轴
        #         axs[3].set_title('Error', fontdict={'family': 'Times New Roman', 'fontsize': 16})
        #
        #         # 绘制凸字形
        #         for ax in axs:
        #             # 绘制四条竖线 起始点是 (4, 2)，结束点是 (4, 8)，表示从 y=2 到 y=8 的竖线。
        #             ax.add_line(plt.Line2D([12, 12], [0, 13], color='yellow', linewidth=2, linestyle='--'))  # 左竖线
        #             ax.add_line(plt.Line2D([36, 36], [0, 13], color='yellow', linewidth=2, linestyle='--'))  # 左竖线
        #             ax.add_line(plt.Line2D([9, 9], [13, 17], color='yellow', linewidth=2, linestyle='--'))  # 左竖线
        #             ax.add_line(plt.Line2D([39, 39], [13, 17], color='yellow', linewidth=2, linestyle='--'))  # 左竖线
        #             ax.add_line(plt.Line2D([9, 12], [13, 13], color='yellow', linewidth=2, linestyle='--'))  # 左竖线
        #             ax.add_line(plt.Line2D([36, 39], [13, 13], color='yellow', linewidth=2, linestyle='--'))  # 左竖线
        #             ax.add_line(plt.Line2D([12, 36], [0, 0], color='yellow', linewidth=2, linestyle='--'))  # 左竖线
        #             ax.add_line(plt.Line2D([9, 39], [17, 17], color='yellow', linewidth=2, linestyle='--'))  # 左竖线
        #
        #         # 创建一个共用的颜色条，为前三个图添加颜色条
        #         cbar1 = fig.colorbar(axs[0].get_images()[0], ax=axs[:3], shrink=0.6)
        #         # 为第四个图添加单独的颜色条
        #         cbar2 = fig.colorbar(axs[3].get_images()[0], ax=axs[3], shrink=0.6)
        #
        #         # 显示图像
        #         plt.show()

    # # 画概率图
    # # 计算误差
    # error_set = torch.abs(true_set[:, :, :, :] - output_set[:, :, :, :])
    # error_values = error_set.numpy().flatten()  # 将误差转换为1D数组
    #
    # # 创建图像
    # fig, ax1 = plt.subplots(figsize=(8, 6))
    #
    # # 绘制误差的柱形图 (直方图)
    # counts, bins, patches = ax1.hist(error_values, bins=100, density=True, alpha=0.6,
    #                                  color=(0 / 255, 128 / 255, 255 / 255))
    # ax1.set_xlabel('Error level', fontdict={'family': 'Times New Roman', 'fontsize': 16})
    # ax1.set_ylabel('Probability density', fontdict={'family': 'Times New Roman', 'fontsize': 16})
    # # ax1.set_title('Error Distribution and Smoothed Curve', fontdict={'family': 'Times New Roman', 'fontsize': 16})
    #
    # # 创建第二个y轴用于绘制平滑曲线图
    # ax2 = ax1.twinx()
    #
    # # 使用核密度估计 (KDE) 来绘制平滑的曲线
    # kde = gaussian_kde(error_values)
    # smooth_x = np.linspace(min(error_values), max(error_values), 500)
    # smooth_y = kde(smooth_x)
    #
    # # 仅绘制概率大于等于0.01的部分
    # mask = smooth_y >= 0.01
    # ax2.plot(smooth_x[mask], smooth_y[mask], color=(255 / 255, 0 / 255, 0 / 255), linewidth=2)
    #
    # # 设置y轴范围
    # ax2.set_ylim(0, max(smooth_y[mask]) * 1.1)
    # # 设置y轴范围
    # ax2.set_xlim(0, max(smooth_x[mask]))
    #
    # # # 设置图例
    # # ax1.legend(loc='upper left')
    # # ax2.legend(loc='upper right')
    #
    # # 显示图像
    # plt.show()

# # 计算均方差的平均值和方差
# mean_mse = np.mean(mse_values)
# variance_mse = np.var(mse_values)

# mean_mae = np.mean(mae_values)
# variance_mae = np.var(mae_values)
# mean_psd = np.mean(psd_values)
# variance_psd = np.var(psd_values)
# mean_gof = np.mean(gof_values)
# variance_gof = np.var(gof_values)
# mean_pde = np.mean(pde_values)
# variance_pde = np.var(pde_values)

# # 输出结果
# print(f"mse均方差的平均值：{mean_mse}")
# print(f"mse均方差的方差：{variance_mse}")

# print(f"mae均方差的平均值：{mean_mae}")
# print(f"mae均方差的方差：{variance_mae}")
# print(f"psd均方差的平均值：{mean_psd}")
# print(f"psd均方差的方差：{variance_psd}")
# print(f"gof均方差的平均值：{mean_gof}")
# print(f"gof均方差的方差：{variance_gof}")
# print(f"pde均方差的平均值：{mean_pde}")
# print(f"pde均方差的方差：{variance_pde}")

# # 找到最小值
# min_value = min(mse_values)
# # 找到所有最小值的索引
# min_indices = [i+1 for i, value in enumerate(mse_values) if value == min_value]
# print(f"The minimum value is {min_value} at Version_{min_indices}.")
# print(f"The length of mse_list is ", len(mse_values))








# 记录结束时间
end_time = time.time()
# 计算代码的运行时间
run_time = end_time - start_time
print("代码运行时间：", run_time, "秒")
