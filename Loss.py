from typing import List, Callable
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F


def get_max_num_cross_loss(num_classes: int):
    num_classes -= 1
    return F.cross_entropy(torch.tensor([1.0, *[0.0] * num_classes]), torch.tensor([*[0.0] * num_classes, 1.0]))


max_2cross_loss = get_max_num_cross_loss(2)
max_3cross_loss = get_max_num_cross_loss(3)
max_6cross_loss = get_max_num_cross_loss(6)
max_9cross_loss = get_max_num_cross_loss(9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
algorithm_turning_point_factor = torch.tensor([0.1183, 0.096, 0.096]).to(device)
algorithm_threshold = torch.tensor([6.0, 10.0, 10.0]).to(device)


def general_loss(outputs: torch.Tensor, labels: torch.Tensor):
    o2 = F.softmax(outputs[6:8])
    o3 = F.softmax(outputs[8:10])
    o4 = F.softmax(outputs[10:12])
    o5 = F.softmax(outputs[12:14])
    o7 = F.softmax(outputs[20:29])
    o8 = F.softmax(outputs[29:32])

    loss1 = F.cross_entropy(outputs[:6], labels[:6]) / max_6cross_loss  # HitFrame
    loss2 = F.cross_entropy(o2, labels[6:8]) / max_2cross_loss  # Hitter
    loss3 = F.cross_entropy(o3, labels[8:10]) / max_2cross_loss  # Roundhand
    loss4 = F.cross_entropy(o4, labels[10:12]) / max_2cross_loss  # Backhand
    loss5 = F.cross_entropy(o5, labels[12:14]) / max_2cross_loss  # BallHeight
    loss7 = F.cross_entropy(o7, labels[20:29]) / max_9cross_loss  # BallType
    loss8 = F.cross_entropy(o8, labels[29:32]) / max_3cross_loss  # Winner

    o6 = outputs[14:20]
    l6 = labels[14:20]
    dist6 = F.pairwise_distance(torch.stack([o6[0::2], o6[1::2]]).permute(1, 0), torch.stack([l6[0::2], l6[1::2]]).permute(1, 0))
    # loss6 = F.mse_loss( torch.stack([o6[0::2], o6[1::2]]).permute(1, 0),torch.stack([l6[0::2], l6[1::2]]).permute(1, 0), reduction = 'none')

    algorithm_6 = dist6 // algorithm_threshold
    algorithm_6 = algorithm_6 // (algorithm_6 - 0.00001)

    loss6 = torch.log10(dist6 + 1) * (
        algorithm_6 * 0.384 + (1 - algorithm_6) * algorithm_turning_point_factor
    )  # Landing, HitterLocation, DefenderLocation

    return (
        0.5 * loss1
        + 0.0625 * loss2
        + 0.03125 * loss3
        + 0.03125 * loss4
        + 0.0625 * loss5
        + 0.0625 * torch.sum(loss6[:2])
        + 0.03125 * torch.sum(loss6[2:])
        + 0.125 * loss7
        + 0.0625 * loss8
    )


def miss_loss(outputs: torch.Tensor, labels: torch.Tensor):
    algor = torch.round(outputs[5])
    return (1.0 - algor) + algor * (0.5 * F.cross_entropy(outputs[:6], labels[:6]) / max_6cross_loss)


class CustomLoss(nn.modules.loss._WeightedLoss):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss_func_ls: List[Callable] = [general_loss, miss_loss]

    def forward(self, outputs, labels):
        loss = 0.0
        i = 0
        with torch.no_grad():
            for output, label in zip(outputs, labels):
                output[:6] = F.softmax(output[:6]).clone()
                loss += self.loss_func_ls[int(label[5])](output, label)
                i += 1
            loss /= i

        return torch.tensor(loss * 100, requires_grad=True)
