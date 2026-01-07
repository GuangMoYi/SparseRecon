import math
import torch
import torch.nn as nn
from einops import rearrange, repeat



# 这个函数通过将位置信息映射到频率空间，使用正弦和余弦函数生成位置编码。
# 这种位置编码可以用于增强神经网络对输入序列的位置信息的感知。
# image_shape表示输入图像的形状，num_frequency_bands表示频率带的数量，max_frequencies表示每个维度的最大频率
def PositionalEncoder(image_shape,num_frequency_bands,max_frequencies=None):
    #  从image_shape中提取出空间维度的形状，忽略通道维度（用下划线 _ 表示）。
    *spatial_shape, _ = image_shape

    # 生成每个空间维度上的坐标，范围从-1到1，步长为相应维度的大小。
    # 第一个张量包含了从 -1 到 1 的 3120 个均匀分布的值，第二个张量包含了从 -1 到 1 的 3 个均匀分布的值。
    # 使用坐标生成位置张量pos，通过torch.meshgrid创建多维网格，然后用torch.stack将网格堆叠起来。
    coords = [ torch.linspace(-1, 1, steps=s) for s in spatial_shape ]
    pos = torch.stack(torch.meshgrid(*coords), dim=len(spatial_shape)) 
    
    encodings = []
    # 如果未提供最大频率，将其设置为位置张量的维度减去最后一维（通道维度）
    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    # 为每个维度生成频率，范围从1.0到最大频率的一半，数量为num_frequency_bands。
    frequencies = [ torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
                                              for max_freq in max_frequencies ]
    
    frequency_grids = []
    # 对每个维度的频率进行计算，得到频率网格，并将其添加到列表中。
    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(pos[..., i:i+1] * frequencies_i[None, ...])

    # 计算正弦值，并将结果扩展到编码列表中。
    #  计算余弦值，并将结果扩展到编码列表中。
    # 将所有编码连接成一个张量，沿最后一个维度（通道维度）拼接。
    # 重新排列张量的维度，将通道维度移动到最后。
    # 返回位置编码的张量。
    encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    enc = torch.cat(encodings, dim=-1)
    enc = rearrange(enc, "... c -> (...) c")

    return enc






