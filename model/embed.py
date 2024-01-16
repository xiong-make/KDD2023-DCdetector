import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """

        Parameters
        ----------
        d_model: 位置编码的维度，即输出的维度
        max_len: 序列的最大长度，用于确定位置编码的最大范围
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # 创建了一个形状为 (max_len, d_model) 的零张量 pe，表示位置编码的矩阵
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        # 创建了表示位置的序列 position，其中 torch.arange(0, max_len) 表示从 0 到 max_len-1 的序列，
        # .float().unsqueeze(1) 将其形状变为 (max_len, 1)。
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 计算位置编码权重 (d_model/2, )
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 最终将位置编码矩阵添加到一个维度为 (1, max_len, d_model) 的新维度上
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# 一维卷积
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            # 参数初始化
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # (batch_size, seq_len, d_model)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x1 (128, 105, 256)
        x1 = self.value_embedding(x)
        # x2 (1, 105, 256)
        x2 = self.position_embedding(x)
        x = x1 + x2
        return self.dropout(x)
