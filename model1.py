import sys, os

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from swin import SwinTransformer
from resnet import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7 * 7
        dim_feedforward = 512
        dropout = 0.1
        num_layers = 6

        self.base_model = resnet18(True, maps=maps)

        self.swin = SwinTransformer(
            hidden_dim=128,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=3,
            num_classes=50,
            head_dim=32,
            window_size=7,
            downscaling_factors=(4, 2, 2, 2),
            relative_pos_embedding=True
        )


        self.lstm = nn.LSTM(dim_feature, dim_feature, 2, bidirectional=True, batch_first=True)

        self.feed = nn.Linear(2 * maps, 2)

        self.feed2 = nn.Linear(82, 2)

    def forward(self, input):

        # --------------------- Transformer ---------------------
        tr_feature = self.swin(input)
        # --------------------- Transformer ---------------------


        # --------------------- ResNet ---------------------
        feature = self.base_model(input)
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        # --------------------- ResNet ---------------------

        # --------------------- LSTM ---------------------
        lstm_feature, _ = self.lstm(feature)
        lstm_feature = lstm_feature[:, :, -1]
        # --------------------- LSTM ---------------------

        all_feature = torch.cat([tr_feature, lstm_feature], 1)
        gaze = self.feed2(all_feature)

        return gaze
