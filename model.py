import sys, os

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy


from resnet import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos

    def forward(self, src):
        # src_mask: Optional[Tensor] = None,
        # src_key_padding_mask: Optional[Tensor] = None):
        # pos: Optional[Tensor] = None):

        # q = k = self.pos_embed(src, pos)

        #### q.size() = (batch_size, 1, feature_dims)
        q = src[:,-1,:].unsqueeze(1)
        k = src
        attn_featrue = self.self_attn(q, k, value=src)[0]
        src = src[:,-1,:].unsqueeze(1) + self.dropout1(attn_featrue)
        # src = self.dropout1(attn_featrue)
        src = src.squeeze(1)
        src = self.norm1(src)

        src = self.linear2(self.activation(self.linear1(src)))
        # src = src + self.dropout2(linear_src)
        # src = self.norm2(src)

        #### q.size() = (batch_size, sequence_len, feature_dims)
        # q = src
        # k = src
        # attn_featrue = self.self_attn(q, k, value=src)[0]
        # src = src + self.dropout1(attn_featrue)
        # # src = self.norm1(src)
        # src = self.linear2(self.activation(self.linear1(src)))

        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7 * 7
        dim_feedforward = 512
        dropout = 0.1
        num_layers = 1

        self.base_model = resnet18(True, maps=maps)

        encoder_layer = TransformerEncoderLayer(maps * dim_feature, nhead, dim_feedforward, dropout)

        encoder_norm = nn.LayerNorm(maps * dim_feature)
        # num_encoder_layer: deeps of layers

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        # self.pos_embedding = nn.Embedding(dim_feature + 1, maps)

        self.lstm = nn.LSTM(maps * dim_feature, maps * dim_feature, 2, bidirectional=False, batch_first=True)

        self.linear1 = nn.Linear(maps * dim_feature, maps * dim_feature)
        self.norm1 = nn.LayerNorm(maps * dim_feature)


        self.feed = nn.Linear(maps * dim_feature, 2)

        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)

        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        # print(input.size())
        batch_size = input.size(0)
        seq_num = input.size(1)
        input = torch.flatten(input, start_dim=0, end_dim=1)
        feature = self.base_model(input)
        feature = feature.flatten(1)
        # feature = self.linear1(feature)

        feature = self.norm1(self.activation(self.linear1(feature)))

        feature = torch.reshape(feature, (batch_size, seq_num, feature.size(1)))


        # --------------------- LSTM ---------------------
        lstm_feature, _ = self.lstm(feature)
        # lstm_feature = lstm_feature[-1, :]
        # --------------------- LSTM ---------------------

        # --------------------- Transformer ---------------------
        # tr_feature = feature.permute(2, 0, 1)  # HW * B * C

        # cls = self.cls_token.repeat((1, batch_size, 1))
        # tr_feature = torch.cat([cls, tr_feature], 0)

        # position = torch.from_numpy(np.arange(0, 50)).to(device)

        # pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        # tr_feature = self.encoder(tr_feature, pos_feature)
        tr_feature = self.encoder(lstm_feature, None)

        # tr_feature = tr_feature.permute(1, 2, 0)

        # tr_feature = tr_feature[:, -1,:]
        # --------------------- Transformer ---------------------

        # all_feature = torch.cat([tr_feature, lstm_feature], 1)
        # gaze = self.feed(all_feature)
        gaze = self.feed(tr_feature)

        return gaze
