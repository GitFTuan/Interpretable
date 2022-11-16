import math
import torch
import torch.nn as nn


class MLTPModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
        super(MLTPModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout = dropout
        self.fan_epoch = fan_epoch
        self.num_heads = num_heads
        self.max_pool = max_pool
        if max_pool == 2:
            shape = 6016
        elif max_pool == 3:
            shape = 3968
        elif max_pool == 4:
            shape = 2944
        elif max_pool == 5:
            shape = 2304
        else:
            shape = 1920
        self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout)
        self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=2,
                                     stride=1
                                     )
        self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1
                                     )
        self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=1
                                     )
        self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=5,
                                     stride=1
                                     )
        # 序列最短为5，故将卷积核分别设为：2、3、4、5
        self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool)

        self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
        self.fan = FAN_encode(self.dropout, shape)

        self.full3 = nn.Linear(shape, 1000)
        self.full4 = nn.Linear(1000, 500)
        self.full5 = nn.Linear(500, 256)
        # self.full6 = nn.Linear(4608, 2304)
        self.Flatten = nn.Linear(256, 64)
        self.out = nn.Linear(64, self.output_size)
        self.dropout = torch.nn.Dropout(self.dropout)

    def TextCNN(self, x):
        x1 = self.conv1(x)
        x1 = torch.nn.ReLU()(x1)
        x1 = self.MaxPool1d(x1)

        x2 = self.conv2(x)
        x2 = torch.nn.ReLU()(x2)
        x2 = self.MaxPool1d(x2)

        x3 = self.conv3(x)
        x3 = torch.nn.ReLU()(x3)
        x3 = self.MaxPool1d(x3)

        x4 = self.conv4(x)
        x4 = torch.nn.ReLU()(x4)
        x4 = self.MaxPool1d(x4)

        y = torch.cat([x1, x2, x3, x4], dim=-1)
        x = self.dropout(y)

        x = x.view(x.size(0), -1)

        return x

    def forward(self, train_data, valid_lens):
        
        embed_output = self.embed(train_data)

        '''----------------------位置编码------------------------'''
        pos_output = self.pos_encoding(self.embed(train_data) * math.sqrt(self.embedding_size))
        '''-----------------------------------------------------'''

        '''----------------------attention----------------------'''
        attention_output = self.attention_encode(pos_output)
        '''-----------------------------------------------------'''

        '''----------------------特征相加-------------------------'''
        vectors = embed_output + attention_output
        '''------------------------------------------------------'''

        '''---------------------data_cnn-----------------------'''
        cnn_input = vectors.permute(0, 2, 1)
        cnn_output = self.TextCNN(cnn_input)
        # x1 = self.conv1(cnn_input)
        # x1 = torch.nn.ReLU()(x1)
        # x1 = self.MaxPool1d(x1)
        # x = self.dropout(x1)
        # cnn_output = x.view(x.size(0), -1)
        '''-----------------------------------------------------'''

        '''---------------------fan_encode----------------------'''
        fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
        for i in range(self.fan_epoch):
            fan_encode = self.fan(fan_encode)
        '''-----------------------------------------------------'''
        # out = cnn_output
        out = fan_encode.squeeze()
        # ab = self.OutLayer(out)

        # 全连接层
        label = self.full3(out)
        label = torch.nn.ReLU()(label)
        label = self.full4(label)
        label = torch.nn.ReLU()(label)
        label = self.full5(label)
        label = torch.nn.ReLU()(label)
        label = self.Flatten(label)
        label = torch.nn.ReLU()(label)
        out_label = self.out(label)

        return out_label


class AddNorm(nn.Module):
    """残差连接后进行层归一化"""

    def __init__(self, normalized, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized)

    def forward(self, X, y):
        return self.ln(self.dropout(y) + X)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈⽹络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建⼀个⾜够⻓的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2,
                                                                                                      dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class AttentionEncode(nn.Module):
    def __init__(self, dropout, embedding_size, num_heads):
        super(AttentionEncode, self).__init__()
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.at1 = nn.MultiheadAttention(embed_dim=self.embedding_size,
                                         num_heads=num_heads,
                                         dropout=0.6
                                         )
        # self.at1 = MultiHeadAttention(key_size=self.embedding_size,
        #                               query_size=self.embedding_size,
        #                               value_size=self.embedding_size,
        #                               num_hiddens=self.embedding_size,
        #                               num_heads=self.num_heads,
        #                               dropout=0.6)
        self.addNorm = AddNorm(normalized=[50, self.embedding_size], dropout=self.dropout)

        self.FFN = PositionWiseFFN(ffn_num_input=64, ffn_num_hiddens=192, ffn_num_outputs=64)

    def forward(self, x, y=None):
        Multi, _ = self.at1(x, x, x)
        # Multi = self.at1(x, x, x, y)
        Multi_encode = self.addNorm(x, Multi)

        # encode_output = self.addNorm(Multi_encode, self.FFN(Multi_encode))

        return Multi_encode


class FAN_encode(nn.Module):
    def __init__(self, dropout, shape):
        super(FAN_encode, self).__init__()
        self.dropout = dropout
        self.addNorm = AddNorm(normalized=[1, shape], dropout=self.dropout)
        self.FFN = PositionWiseFFN(ffn_num_input=shape, ffn_num_hiddens=(2*shape), ffn_num_outputs=shape)

    def forward(self, x):
        encode_output = self.addNorm(x, self.FFN(x))

        return encode_output


def sequence_mask(X, valid_len, value=0.):
    """在序列中屏蔽不相关的项"""
    valid_len = valid_len.float()
    MaxLen = X.size(1)
    mask = torch.arange(MaxLen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None].to(X.device)
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)  # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意⼒"""

    def __init__(self, key_size, query_size, num_hiddens, dropout):
        super(AdditiveAttention, self).__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使⽤⼴播⽅式进⾏求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class net_fan(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout):
        super(net_fan, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout = dropout

        self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)

        self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=2,
                                     stride=1
                                     )

        self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=5)

        self.full5 = nn.Linear(576, 256)
        self.Flatten = nn.Linear(256, 64)
        self.out = nn.Linear(64, self.output_size)
        self.dropout = torch.nn.Dropout(self.dropout)

    def TextCNN(self, x):
        x1 = self.conv1(x)
        x1 = torch.nn.ReLU()(x1)
        x1 = self.MaxPool1d(x1)
        x = self.dropout(x1)

        x = x.view(x.size(0), -1)
        return x

    def forward(self, train_data):
        embed_output = self.embed(train_data)

        '''---------------------data_cnn-----------------------'''
        cnn_input = embed_output.permute(0, 2, 1)
        cnn_output = self.TextCNN(cnn_input)
        '''-----------------------------------------------------'''

        out = cnn_output.squeeze()

        # 全连接层
        label = self.full5(out)
        label = torch.nn.ReLU()(label)
        label = self.Flatten(label)
        label = torch.nn.ReLU()(label)
        out_label = self.out(label)

        return out_label


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(attention_weights), values)
