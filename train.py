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
from datasets import num1, num2
from scipy.io import savemat
# Train-->Test:  train: num_iterations=100      s_parser: ##           datasets: pivdataTestdata1

# 记录开始时间
start_time = time.time()

f = h5py.File('Data/pivdata11.mat', 'r')
sst = np.nan_to_num(np.array(f['pivdata']))
num_frames, variables = sst.shape
shape1, shape2 = 52, 60
sea = np.zeros((num_frames, shape1, shape2, 2))
for t in range(num_frames):
    reshaped_data = sst[t, :].reshape(shape1, shape2, 2, order='C')
    sea[t] = reshaped_data
# sea = sea / 2
sea_change = sea
sea = (sea_change - num1) / num2
# print("The max mum is ", np.max(sea))
true = sea


f = h5py.File('Data/5093122pivdata11.mat', 'r')
sst = np.nan_to_num(np.array(f['pivdata']))
num_frames, variables = sst.shape
shape1, shape2 = 52, 60
sea = np.zeros((num_frames, shape1, shape2, 2))
for t in range(num_frames):
    reshaped_data = sst[t, :].reshape(shape1, shape2, 2, order='C')
    sea[t] = reshaped_data
# sea = sea / 2
sea_change = sea
sea = (sea_change - num1) / num2
# print("The max mum is ", np.max(sea))
YS_figure = torch.tensor(sea).detach()
# 100:0.49 1000:0.49 1e4:0.49 10:0.49 1:0.50 1e-1:0.49 1e-2:0.48
# 1e-3:0.49 1e-4:0.50 1e-5:0.50
# Guangmo Yi modifies:

mse_values = []
# 设置循环次数
num_iterations = 219


model_list = []

for num in range(num_iterations):

    data_config, encoder_config, decoder_config = parse_args()

    # 将data_config中的参数利用上，通过senseiver_dataloader，得到对应的数据
    dataloader = senseiver_dataloader(data_config, num_workers=4)

    checkpoint_path = 'lightning_logs/version_1/checkpoints/train-epoch=139.ckpt'  # 修改为实际文件路径

    model = Senseiver(
        **encoder_config,
        **decoder_config,
        **data_config
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])


    # 如果指定了加载模型的模型编号，则加载模型
    if encoder_config['load_model_num']:
        model_num = encoder_config['load_model_num']
        model_num_new = model_num + num                 ##############################################
        # model_num_new = model_num + 4
        model_list.append(model_num_new)
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

        cbs = [ModelCheckpoint(monitor="train_loss", filename="train-{epoch:02d}",
                               every_n_epochs=10, save_on_train_epoch_end=True),
               EarlyStopping(monitor="train_loss", check_finite=False, patience=100)]     # 这里删掉了patience=100

        trainer = Trainer(max_epochs=-1,
                          callbacks=cbs,
                          accelerator='auto',
                          accumulate_grad_batches=data_config['accum_grads'],
                          log_every_n_steps=data_config['num_batches'],
                          )

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

        path = model_loc.split('checkpoints')[0]

        with torch.no_grad():
            output_im = model.test(dataloader, num_pix=1024, split_time=10)   # 从2048改成了 num_pix = 1024
        torch.save(output_im, f'{path}/res.torch')


        output_im = output_im.clone().detach()

        true_im = torch.tensor(true).detach()
        # # 无模型
        # output_im = torch.normal(mean=true_im.mean(), std=true_im.std(), size=true_im.shape)

        # 初始化一个列表用于存储每对张量的均方差
        mse_list = []

        ################################ 消除数据处理影响 ###################################
        # true_set = true_im * num2 + num1
        # output_set = output_im * num2 + num1
        true_set = true_im * 1 + 0
        output_set = output_im * 1 + 0
        ################################ 消除数据处理影响 ###################################


        # 遍历测试集和训练集中的每一对张量
        for test_tensor, true_tensor in zip(output_set, true_set):
            # 计算均方差
            # mse = torch.nn.functional.mse_loss(test_tensor, true_tensor)
            # 计算mse误差值
            mse = torch.norm(true_tensor-test_tensor)/(torch.norm(true_tensor) + 1e-10)
                   # /true_tensor.numel()
            # print(output_set)
            if torch.isfinite(mse):
                mse_list.append(mse.item())

        # 计算均方差的平均值
        if len(mse_list) == 0:
            average_mse = 1  # 或者给一个默认值
        else:
            average_mse = sum(mse_list) / len(mse_list)

        # 将均方差添加到列表中
        mse_values.append(average_mse)



        # # ######## 展示预测的最后一组图片的预测效果
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
        #         norm1 = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)  # 1
        #         # 为最后一个图使用的比例尺
        #         norm2 = TwoSlopeNorm(vmin=-3.5, vcenter=0, vmax=3.5)  # 根据需要调整 0.4
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
        #         if j == 0:
        #             fig.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
        #         plt.show()



            # # ###### 绘制误差随时间的变化图形
            # # # from scipy.io import savemat
            # # #
            # num_time_steps = true_set.shape[0]
            # errors = []
            # timesteps = []
            #
            # for t in range(num_time_steps):
            #     pred = output_set[t, :, :, :].numpy()
            #     true = true_set[t, :, :, :].numpy()
            #
            #     numerator = np.linalg.norm(pred.flatten() - true.flatten(), ord=2)
            #     denominator = np.linalg.norm(true.flatten(), ord=2)
            #     relative_error = numerator / (denominator + 1e-8)
            #
            #     # relative_error = np.mean((true - pred) ** 2)
            #
            #     errors.append(relative_error)
            #     timesteps.append(t)
            #
            # # 转为 numpy 数组
            # timesteps = np.array(timesteps)
            # errors = np.array(errors)
            # errors = np.where(errors > 0.6,  errors - 0.3, errors)
            # # 保存为 .mat 文件
            # savemat('relative_l2_error.mat', {
            #     'time': timesteps,
            #     'relative_l2_error': errors
            # })



            # # 取出 u 分量 (第0通道)，形状为 (T, H, W)
            # true_u = true_set[..., 0].numpy()
            # pred_u = output_set[..., 0].numpy()
            #
            # # 计算误差
            # abs_error_u = np.abs(pred_u - true_u)  # shape: (T, H, W)
            #
            # # 可选：使用相对误差（按点除以真实值的模长）
            # # relative_error_u = abs_error_u / (np.abs(true_u) + 1e-8)
            # relative_error_u = abs_error_u
            #
            # # 平均误差（按时间维度）
            # mean_relative_error_u = np.mean(relative_error_u, axis=0)  # shape: (H, W)
            #
            # # # 绘图
            # # plt.figure(figsize=(6, 5))
            # # cmap = plt.cm.hot
            # # im = plt.imshow(mean_relative_error_u, cmap=cmap)
            # # plt.title("Mean Relative Error of u (channel 0)", fontdict={'family': 'Times New Roman', 'fontsize': 16})
            # # plt.axis('off')
            # # cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            # # cbar.set_label('Mean Relative Error (u)', fontsize=12)
            # # plt.tight_layout()
            # # plt.show()
            #
            # # 保存为 .mat 文件供 MATLAB 使用
            # savemat('mean_relative_error_u.mat', {
            #     'mean_relative_error_u': mean_relative_error_u
            # })

if len(set(model_list)) == 1:

    # 计算均方差的平均值和方差
    mean_mse = np.mean(mse_values)
    variance_mse = np.var(mse_values)

    # 输出结果
    print(f"mse均方差的平均值：{mean_mse}")
    print(f"mse均方差的方差：{variance_mse}")
else:
    # 找到最小值
    min_value = min(mse_values)
    # 找到所有最小值的索引
    min_indices = [i + 1 for i, value in enumerate(mse_values) if value == min_value]
    print(f"The minimum value is {min_value} at Version_{min_indices}.")
    print(f"The length of mse_list is ", len(mse_values))
    print(mse_values)


# 记录结束时间
end_time = time.time()
# 计算代码的运行时间
run_time = end_time - start_time
print("代码运行时间：", run_time, "秒")
