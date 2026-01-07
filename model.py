import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from fairscale.nn import checkpoint_wrapper


# The code to build the model is modified from:
# https://github.com/krasserm/perceiver-io


# 用于构建顺序执行多个模块的模型
class Sequential(nn.Sequential):
    # 重写了前向传播方法 forward
    def forward(self, *inputs):
        for module in self:
            # 如果是元组，则将元组打开，作为单独参数输入于module，否则直接作为参数输入
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def mlp(num_channels: int, dropout=0.1):
    return Sequential(
        Residual(nn.LayerNorm(num_channels), dropout),
        Residual(nn.Linear(num_channels, num_channels), dropout),
        nn.GELU(),
        Residual(nn.Linear(num_channels, num_channels), dropout),
    )

# def mlp(num_channels: int, dropout=0.1, num_hidden_layers=2, hidden_dim_multiplier=3):
#     layers = [Residual(nn.LayerNorm(num_channels), dropout)]
#     for _ in range(num_hidden_layers):
#         layers.append(nn.Linear(num_channels, num_channels * hidden_dim_multiplier))
#         layers.append(nn.GELU())
#         layers.append(nn.Linear(num_channels * hidden_dim_multiplier, num_channels))
#     return Sequential(*layers)


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=None, dropout=0.1):
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers if num_layers is not None else 1
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, input_size)
        # self.activate = nn.ELU(alpha=1.0)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out = self.activate(out)
        out = self.linear(out)
        out = self.dropout(out)
        return out


def cross_attention_layer( num_q_channels: int,
                           num_kv_channels: int,
                           num_heads: int,
                           dropout: float,
                           activation_checkpoint: bool = False
                           ):
    layer = Sequential(
        Residual(CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout), dropout),
        Residual(MyRNN(num_q_channels, num_q_channels), dropout),
        Residual(mlp(num_q_channels), dropout),
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_layer(num_channels: int,
                         num_heads: int,
                         dropout: float,
                         activation_checkpoint: bool = False,
                         ):

    layer = Sequential(
        Residual(SelfAttention(num_channels, num_heads, dropout), dropout),
        Residual(mlp(num_channels), dropout),
        Residual(MyRNN(num_channels, num_channels), dropout),
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_block(num_layers: int,
                         num_channels: int,
                         num_heads: int,
                         dropout: float,
                         activation_checkpoint: bool = False):

    # 通过循环num_layers次，创建一个包含多个自注意力层的列表
    layers = [self_attention_layer(
                            num_channels,
                            num_heads,
                            dropout,
                            activation_checkpoint) for _ in range(num_layers)]

    # *layers 会将列表 layers 中的所有元素作为参数传递给 Sequential 构造函数，
    # 相当于将多个自注意力层模块按顺序加入到了这个序列模块中。
    return Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]




class MultiHeadAttention(nn.Module):
    # 注意力头的数量 num_heads 和丢弃率 dropout。
    def __init__(self, num_q_channels: int,
                 num_kv_channels: int,
                 num_heads: int,
                 dropout: float):

        super().__init__()

        # 创建一个多头注意力层，使用 PyTorch 内置的 nn.MultiheadAttention。
        # embed_dim=num_q_channels: 设置查询（Q）、键（K）、值（V）的嵌入维度为查询（Q）通道数。
        # batch_first=True: 表示输入的第一个维度是批次大小，而不是序列长度
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    # 可选的填充掩码 pad_mask 和注意力掩码 attn_mask
    # pad_mask: 一个可选的张量，用于指定哪些元素需要被忽略。在注意力计算中，这可以用于屏蔽填充部分的序列。
    # attn_mask: 一个可选的张量，用于指定哪些位置的注意力需要被屏蔽。这在序列生成任务中可能很有用。
    # nn.MultiheadAttention 的前向传播方法返回一个元组，其中包含注意力张量和注意力权重。我们这里只关心注意力张量，因此通过 [0] 获取它
    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask, attn_mask=attn_mask)[0]


# CrossAttention模块实现了交叉注意力机制：通过对查询和键值进行归一化，并将归一化后的张量传递给多头注意力层进行处理
class CrossAttention(nn.Module):
    # num_heads: 注意力头的数量。
    def __init__(self,
                 num_q_channels: int,
                 num_kv_channels: int,
                 num_heads: int,
                 dropout: float):

        # self.q_norm: 创建一个查询通道的层归一化（LayerNorm）层。用于后续对x_q进行标准化
        # self.kv_norm: 创建一个键值通道的层归一化层。
        # self.attention: 创建一个多头注意力层，使用了之前定义的MultiHeadAttention模块
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(
                                            num_q_channels=num_q_channels,
                                            num_kv_channels=num_kv_channels,
                                            num_heads=num_heads,
                                            dropout=dropout
                                            )

    # pad_mask: 可选的填充掩码，默认为None。 attn_mask: 可选的注意力掩码，默认为None。
    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


# 和CrossAttention类似
class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
                                            num_q_channels=num_channels,
                                            num_kv_channels=num_channels,
                                            num_heads=num_heads,
                                            dropout=dropout
                                            )

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


# 定义了一个Encoder类

class MyCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        # dropout = 0.1
        # 定义卷积层、批量归一化层和池化层，使用动态计算的通道数
        self.conv1_block = Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1),
            # nn.BatchNorm1d(input_channels),
            nn.SiLU(),
            # nn.AvgPool1d(kernel_size=3, stride=1, padding=1)  # 保持序列长度
        )

        self.conv2_block = Sequential(
            nn.Conv1d(in_channels=input_channels * 2, out_channels=input_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_channels * 4),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)  # 保持序列长度
        )

        self.conv3_block = Sequential(
            nn.Conv1d(in_channels=input_channels * 4, out_channels=input_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_channels * 8),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)  # 保持序列长度
        )

        # 使用1x1卷积确保通道数一致
        self.conv1_residual = nn.Conv1d(input_channels, input_channels * 2, kernel_size=1)
        self.conv2_residual = nn.Conv1d(input_channels * 2, input_channels * 4, kernel_size=1)
        self.conv3_residual = nn.Conv1d(input_channels * 4, input_channels * 8, kernel_size=1)

        # 替换全连接层为卷积层，使用动态通道数
        self.linear = nn.Linear(input_channels, output_channels)

        # Dropout层
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 第一个卷积块并应用残差连接
        # residual = self.conv1_residual(x)
        x = self.conv1_block(x)
        # x += residual

        # 第二个卷积块并应用残差连接
        # residual = self.conv2_residual(x)
        # x = self.conv2_block(x)
        # x += residual

        # 第三个卷积块并应用残差连接
        # residual = self.conv3_residual(x)
        # x = self.conv3_block(x)
        # x += residual

        # 将数据转换为线性层所需的形状
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = x.permute(0, 2, 1)

        # 应用Dropout
        x = self.dropout(x)

        return x



# 定义了一个Encoder类
class Encoder(nn.Module):
    # input_ch: 输入通道的数量。
    # preproc_ch: 预处理通道的数量（可选）。
    # num_latents: 潜在表示的数量。
    # num_latent_channels: 每个潜在表示的通道数量。
    # num_layers: 编码器层数，默认为3。
    # num_cross_attention_heads: 交叉注意力层的注意力头数量，默认为4。
    # num_self_attention_heads: 自注意力层的注意力头数量，默认为4。
    # num_self_attention_layers_per_block: 每个块中自注意力层的数量，默认为6。
    # dropout: 丢弃率，默认为0.0。
    # activation_checkpoint: 是否使用激活检查点，默认为False。

    def __init__(
            self,
            input_ch,
            preproc_ch,

            num_latents: int,
            num_latent_channels: int,
            num_layers: int = 3,
            num_cross_attention_heads: int = 4,
            num_self_attention_heads: int = 4,
            num_self_attention_layers_per_block: int = 6,
            dropout: float = 0.1,
            activation_checkpoint: bool = False,
    ):

        super().__init__()

        self.num_layers = num_layers

        # 根据用户是否提供 preproc_ch 的值，来决定是否创建一个额外的线性层作为预处理层
        # 将输入特征维度设为 input_ch，输出特征维度设为 preproc_ch
        if preproc_ch:
            # self.norm = nn.LayerNorm(input_ch)                  # can change
            self.preproc = nn.Linear(input_ch, preproc_ch)
            # 调用CNN层替换Linear
            self.preproc = nn.Conv1d(in_channels=input_ch, out_channels=preproc_ch, kernel_size=3, padding=1)
            # self.preproc = MyCNN(input_channels=input_ch, output_channels=preproc_ch)
        else:
            self.preproc = None
            preproc_ch = input_ch

        # 论文模型中的：Attention block，这里的cross_attention_layer是论文模型中的CrossAttention和MLP
        def create_layer():
            return Sequential(
                cross_attention_layer(
                    num_q_channels=num_latent_channels,
                    num_kv_channels=preproc_ch,
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
                ),
                self_attention_block(
                    num_layers=num_self_attention_layers_per_block,
                    num_channels=num_latent_channels,
                    num_heads=num_self_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
                ),
            )

        self.layer_1 = create_layer()
        # torch.empty(num_latents, num_latent_channels): 创建了一个大小为 (num_latents, num_latent_channels) 的未初始化的张量
        # nn.Parameter(...): 将上述创建的张量包装成一个可学习的参数。通过将张量包装成 nn.Parameter，模型在训练过程中可以学习并更新这个参数的数值。
        if num_layers > 1:
            self.layer_n = create_layer()
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        # 对 self.latent 参数进行初始化，从正态分布中抽取随机数，均值为 0，标准差为 0.02
        # 对初始化后的数值进行截断，将小于 -2.0 的数截断为 -2.0，将大于 2.0 的数截断为 2.0。
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape

        # 论文模型中的： E^(1) = Linear[E^(0)]
        if self.preproc:
            # x = self.norm(x)                                 # can change
            # x = self.preproc(x)
            # 调用CNN层替换Linear
            x = x.permute(0, 2, 1)  # 调整形状以匹配Conv1d的输入
            x = self.preproc(x)
            x = x.permute(0, 2, 1)  # 调整回原来的形状

        x_latent = repeat(self.latent, "... -> b ...", b=b)

        # 论文模型中的：E^(2) = Attention block[E^(1)]
        x_latent = self.layer_1(x_latent, x, pad_mask)

        # 论文模型中的：z = Attention block[E^(2)]
        for i in range(self.num_layers - 1):
            x_latent = self.layer_n(x_latent, x, pad_mask)

        return x_latent


class Decoder(nn.Module):
    # ff_channels: 前馈神经网络通道数，指定前馈神经网络的输出通道数。
    # preproc_ch: 预处理通道数（可选），默认可以为 None。如果指定了预处理通道数，则在 Decoder 的结构中会使用预处理线性层。
    # num_latent_channels: 潜在向量通道数，指定潜在向量的通道数。
    # latent_size: 潜在向量大小，指定潜在向量的维度大小。
    # num_output_channels: 输出通道数，指定 Decoder 最终输出的通道数。
    # num_cross_attention_heads: 跨注意力层的注意力头数，默认为 4。
    # activation_checkpoint: 激活检查点，如果为 True，则在激活函数之前使用检查点技术，以减少内存使用。默认为 False
    def __init__(
            self,
            ff_channels: int,
            preproc_ch,
            num_latent_channels: int,
            latent_size,
            num_output_channels,
            num_cross_attention_heads: int = 4,
            dropout: float = 0.1,
            activation_checkpoint: bool = False,
    ):

        super().__init__()
        # 计算查询通道数，将前馈神经网络通道数(network_light.py中定义为=pos_encoder_ch)
        # 和潜在向量通道数(network_light.py中定义为=dec_num_latent_channels)相加。
        q_chan = ff_channels + num_latent_channels  # ff_channels相当于是位置编码维度
        # 如果预处理通道数存在，则将查询通道数 q_in 设置为预处理通道数，否则设置为计算得到的查询通道数 q_chan。
        if preproc_ch:
            q_in = preproc_ch
        else:
            q_in = q_chan

        # 这是用来输出最终的输出的，确保最后的输出的维度与原始输入数据（Sensor value）相同 num_output_channels = 2
        # self.postproc = nn.Linear(q_in, num_output_channels)

        # self.norm2 = nn.BatchNorm1d(q_in)                        # can change
        self.postproc = nn.Conv1d(in_channels=q_in, out_channels=num_output_channels, kernel_size=3, padding=1)
        # self.postproc = MyCNN(input_channels=q_in, output_channels=num_output_channels)
        # 如果预处理通道数存在，则创建一个线性层 self.preproc，将输入特征维度为 q_chan 的数据映射到输出特征维度为 preproc_ch
        if preproc_ch:
            self.norm1 = nn.LayerNorm(q_chan)                      # can change
            self.preproc = nn.Linear(q_chan, preproc_ch)
            # self.preproc = nn.Conv1d(in_channels=q_chan, out_channels=preproc_ch, kernel_size=3, padding=1)
        else:
            self.preproc = None

        # 创建一个跨注意力层，使用定义的 cross_attention_layer 函数。
        self.cross_attention = cross_attention_layer(
            num_q_channels=q_in,
            num_kv_channels=num_latent_channels,
            num_heads=num_cross_attention_heads,
            dropout=dropout,
            activation_checkpoint=activation_checkpoint,
        )

        # 创建一个可学习的参数 self.output，是一个形状为 (latent_size, num_latent_channels) 的张量。  latent_size = 16
        self.output = nn.Parameter(torch.empty(latent_size, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, coords):
        b, *_ = x.shape

        # 用 PyTorch 的 repeat_interleave 函数，将 output 沿着第二个维度（axis=1）重复复制，
        # 重复的次数由 coords.shape[1] 决定。这可能是为了与坐标信息（coords）匹配。
        output = repeat(self.output, "... -> b ...", b=b)
        # 论文模型中的： Q^(out)
        output = torch.repeat_interleave(output, coords.shape[1], axis=1)

        # 论文模型中的： D^(0) = a_q + Q_out
        output = torch.cat([coords, output], axis=-1)

        # 论文模型中的： D^(1) = Linear[D^(0)]
        if self.preproc:
            output = self.norm1(output)                                   # can change
            output = self.preproc(output)
            # output = output.permute(0, 2, 1)
            # output = self.preproc(output)
            # output = output.permute(0, 2, 1)

        # 论文模型中的： D^(2) = CrossAttention[D^(1), kv]
        output = self.cross_attention(output, x)

        # output = self.postproc(output)

        output = output.permute(0, 2, 1)
        # output = self.norm2(output)                                    # can change
        output = self.postproc(output)
        output = output.permute(0, 2, 1)


        return output





