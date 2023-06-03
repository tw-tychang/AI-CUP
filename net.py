import torch
import torch.nn as nn
from net_pretrained_effv2 import EffNet


class LinNet(nn.Module):
    def __init__(self, input, output_classes, isOneHot=False):
        super(LinNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_classes),
        )

        self.last = nn.Softmax() if isOneHot else self.none_func

    @staticmethod
    def none_func(x):
        return x

    def forward(self, x):
        return self.last(self.linear(x))


class BadminationNet(nn.Module):
    def __init__(self, in_seq, output_classes):
        super(BadminationNet, self).__init__()

        eff_out = 2048
        self.eff = EffNet(in_seq=in_seq, output_classes=eff_out)
        self.lins = nn.ModuleList(
            [
                LinNet(eff_out, 6, isOneHot=True),  # 0~6
                LinNet(eff_out, 2, isOneHot=True),  # 6~8
                LinNet(eff_out, 2, isOneHot=True),  # 8~10
                LinNet(eff_out, 2, isOneHot=True),  # 10~12
                LinNet(eff_out, 2, isOneHot=True),  # 12~14
                LinNet(eff_out, 6, isOneHot=False),  # 12~14
                LinNet(eff_out, 9, isOneHot=True),  # 20~29
                LinNet(eff_out, 3, isOneHot=True),  # 29~32
            ]
        )

    def forward(self, x):
        x = self.eff(x)
        return [lin(x) for lin in self.lins]

    def update(self):
        ...
