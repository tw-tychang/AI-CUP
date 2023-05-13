import torch
import torch.nn as nn
import math
from EfficientNet import effnetv2_m


class Conveter(nn.Module):
    def __init__(self, in_seq):
        super(Conveter, self).__init__()

        self.conv = nn.Sequential(
            # pw
            nn.Conv3d(in_seq, 1, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return torch.squeeze(x)


class BadminationNet(nn.Module):
    def __init__(self, in_seq, output_classes):
        super(BadminationNet, self).__init__()
        self.conveter = Conveter(in_seq)
        self.features = effnetv2_m(num_classes=32)
        self.linear = nn.Sequential(
            nn.Linear(self.features.output_channel, 1000),
            nn.Linear(1000, output_classes),
        )

    def forward(self, x):
        x = self.conveter(x)
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.linear(x)
        return x
