import torch
from torch import nn
from torchvision.models.efficientnet import efficientnet_v2_m, EfficientNet_V2_M_Weights
from net import Conveter


def efficientnet_v2_m_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x


class Net(nn.Module):
    def __init__(self, in_seq, output_classes, *args, **kwargs) -> None:
        super(Net, self).__init__(*args, **kwargs)

        self.convert = Conveter(in_seq)
        self.features = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.features.forward = efficientnet_v2_m_forward
        self.linear = nn.Sequential(
            nn.Linear(self.features.lastconv_output_channels, 1000),
            nn.Linear(1000, output_classes),
        )

    def forward(self,x):
        x = self.convert(x)
        x = self.features(x)
        x = self.linear(x)
        return x