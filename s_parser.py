import os
import argparse
import torch
import numpy as np
import h5py

def str2bool(v):
    # 首先检查输入值 v 是否已经是布尔类型。如果是，直接返回该值，不进行进一步的转换。
    if isinstance(v, bool):
        return v
    # 如果输入值的小写形式在指定的字符串集合中（"yes", "true", "t", "y", "1"），则将其转换为布尔值 True。
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



# 用于定义数据、编码器、解码器的配置
def parse_args():
    f = h5py.File('Data/pivdata11.mat', 'r')
    # 使用 np.nan_to_num 函数将数组中的 NaN 值替换为零，python提取数据会将数据倒转一次
    sst = np.nan_to_num(np.array(f['pivdata']))
    num_frames, _ = sst.shape

    # 创建了一个命令行解析器对象，称之为 parser。
    # description="Senseiver" 提供了一个简短描述，说明这个程序是关于什么的。
    # 后面就可以定义程序期望接受的命令行参数的类型和其他信息。
    parser = argparse.ArgumentParser(description="Senseiver")

    # Data
    # parser.add_argument("--data_name", default=None, type=str)
    # Guangmo Yi modifies:
    parser.add_argument("--data_name", default="sea", type=str)
    parser.add_argument("--num_sensors", default=256, type=int)     # 原来是8，表示训练过程随机取8个点进行稀疏重建
    parser.add_argument("--gpu_device", default=0, type=int)
    # 训练的图片数目
    parser.add_argument("--training_frames", default=num_frames, type=int)
    parser.add_argument("--consecutive_train", default=False, type=str2bool)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--batch_frames", default=64, type=int)     # 64
    parser.add_argument("--batch_pixels", default=1024, type=int)    # 由2048改成了1024
    parser.add_argument("--lr", default=0.0001, type=float)          # 学习率，参数更新的幅度 0.0001
    # parser.add_argument("--accum_grads", default=None, type=int)
    # Guangmo Yi modifies:
    # "--accum_grads" = 1，表示每次获得累积梯度1次就更新一次参数
    parser.add_argument("--accum_grads", default=1, type=int)

    # Positional Encodings
    parser.add_argument("--space_bands", default=2, type=int)   # 32 best:2 ->2:0.55 4:0.56 64:0.54  (1) 0.51  --> 2 很好

    # # Train or Test
    parser.add_argument("--load_model_num", default=None, type=int)
    parser.add_argument("--test", default=False, type=str2bool)

    # # Guangmo Yi modifies
    # parser.add_argument("--load_model_num", default=1, type=int)
    # parser.add_argument("--test", default=True, type=str2bool)

    # Encoder
    # 预处理通道数量，用于扩展输入维度
    parser.add_argument("--enc_preproc_ch", default=256, type=int)   # 64 best:256-->512[24] 256:0.56  (2) 256很好
    # 指定编码器中的"seq" latent数量。编码器会产生多个潜在表示，每个表示对应输入序列中的一个部分或特征。
    # 对于piv数据而言，“位置2，速度1，x速度1，y速度1”，可以考虑设置latent数量为5，但是一般取2的倍数
    parser.add_argument("--num_latents",   default=4,   type=int)
    # Latent通道数量，用于指定编码器中的通道数量，设为 16，表示每个潜在表示都是一个 16 维的向，需要和下文的"--dec_num_latent_channels"一致
    parser.add_argument("--enc_num_latent_channels", default=64, type=int)   # 16 best:512-->32:0.54 64:0.51 256:0.52 512:0.45(Ver22)  64
    # 编码器中的层数
    parser.add_argument("--num_layers", default=2, type=int)   # 2
    # 交叉注意力层中的注意力头数量
    parser.add_argument("--num_cross_attention_heads", default=2, type=int)
    # 自注意力层中的注意力头数量
    parser.add_argument("--enc_num_self_attention_heads", default=16, type=int)  # 2 best:2or16--> 2：0.58
    # 每个块中的自注意力层的数量
    parser.add_argument("--num_self_attention_layers_per_block", default=2, type=int) # 3[存疑]-->256[51] 1:0.64 2:0.58 32:0.52(感觉不行) 256：0.53 （试试256）
    # 用于指定dropout层的丢弃率
    parser.add_argument("--dropout", default=0.1, type=float)          # 现在改成了0.05

    # Decoder
    # 预处理通道数量。对 Latent 表示进行一些变换、缩放、调整或其他预处理操作
    parser.add_argument("--dec_preproc_ch", default=None, type=int)
    # Latent通道数量
    parser.add_argument("--dec_num_latent_channels", default=64, type=int)  # 16 同enc_num_latent_channels
    # 交叉注意力层中的注意力头数量
    parser.add_argument("--dec_num_cross_attention_heads", default=1, type=int)  # 1

    # 解析命令行参数，并放到args中
    args = parser.parse_args()

    if torch.cuda.is_available():
        accelerator = "gpu"
        gpus = [args.gpu_device]
    else:
        accelerator = "cpu"
        gpus = None

    # 把前面给的参数组合到data_config这个字典里
    data_config = dict(data_name = args.data_name,
                       num_sensors = args.num_sensors,
                       gpu_device=None if accelerator == 'cpu' else gpus,
                       accelerator = accelerator,
                       training_frames = args.training_frames,
                       consecutive_train = args.consecutive_train,
                       seed = args.seed,
                       batch_frames = args.batch_frames,
                       batch_pixels = args.batch_pixels,
                       lr=args.lr,
                       accum_grads = args.accum_grads,
                       test = args.test,
                       space_bands=args.space_bands,
                       )



    encoder_config = dict(load_model_num=args.load_model_num,
                          enc_preproc_ch=args.enc_preproc_ch,  # 输入变量的特征维度
                          num_latents=args.num_latents,     # "seq" latent
                          enc_num_latent_channels=args.enc_num_latent_channels,  # channels [b,seq,chan]
                          num_layers=args.num_layers,
                          num_cross_attention_heads=args.num_cross_attention_heads,
                          enc_num_self_attention_heads=args.enc_num_self_attention_heads,
                          num_self_attention_layers_per_block=args.num_self_attention_layers_per_block,
                          dropout=args.dropout,
                          )


    decoder_config = dict(dec_preproc_ch=args.dec_preproc_ch,  # latent bottleneck
                          dec_num_latent_channels=args.dec_num_latent_channels,  # hyperparam
                          latent_size=1,  # collapse from n_sensors to 1 observation，表示Decoder的输出为一维
                          dec_num_cross_attention_heads=args.dec_num_cross_attention_heads
                          )


    return data_config, encoder_config, decoder_config