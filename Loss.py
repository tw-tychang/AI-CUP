import numpy
import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss().to(device)
mse = nn.MSELoss().to(device)
max_loss = 2.3

def loss():
    #crossentropy
    loss_c = criterion() 

    #MSE
    loss_m = mse()

    def badmination_loss(outputs, labels):
        if labels[5] == 0:
            loss1 = loss_c(outputs[:6], labels[:6]) / max_loss  # HitFrame
            loss2 = loss_c(outputs[6:8], labels[6:8]) / max_loss  # Hitter
            loss3 = loss_c(outputs[8:10], labels[8:10]) / max_loss  # Roundhand
            loss4 = loss_c(outputs[10:12], labels[10:12]) / max_loss  # Backhand
            loss5 = loss_c(outputs[12:14], labels[12:14]) / max_loss  # BallHeight
            loss6 = torch.log6(
                torch.add(
                    torch.sqrt(
                        torch.add(
                            torch.pow(torch.sub(labels[14], outputs[14]), 2),
                            torch.pow(torch.sub(labels[15], outputs[15]), 2),
                        )
                    )
                ),
                1,
            )  # Landing
            loss7 = torch.log6(
                torch.add(
                    torch.sqrt(
                        torch.add(
                            torch.pow(torch.sub(labels[16], outputs[16]), 2),
                            torch.pow(torch.sub(labels[17], outputs[17]), 2),
                        )
                    )
                ),
                1,
            )  # HitterLocation
            loss8 = torch.log6(
                torch.add(
                    torch.sqrt(
                        torch.add(
                            torch.pow(torch.sub(labels[18], outputs[18]), 2),
                            torch.pow(torch.sub(labels[19], outputs[19]), 2),
                        )
                    )
                ),
                1,
            )  # DefenderLocation
            loss9 = loss_c(outputs[20:29], labels[20:29]) / max_loss  # BallType
            loss10 = loss_c(outputs[29:32], labels[29:32]) / max_loss  # Winner

            total_loss = torch.add(
                torch.mul(0.5, loss1),
                torch.mul(0.0625, loss2),
                torch.mul(0.03125, loss3),
                torch.mul(0.03125, loss4),
                torch.mul(0.0625, loss5),
                torch.mul(0.0625, loss6),
                torch.mul(0.03125, loss7),
                torch.mul(0.03125, loss8),
                torch.mul(0.125, loss9),
                torch.mul(0.0625, loss10),
            )

        elif labels[5] == 1:
            if labels[5] == outputs[5]:
                loss1 = loss_c(outputs[:6], labels[:6]) / max_loss  # HitFrame
                loss2 = loss_m(outputs[6:8], labels[6:8]) / max_loss  # Hitter
                loss3 = loss_m(outputs[8:10], labels[8:10]) / max_loss  # Roundhand
                loss4 = loss_m(outputs[10:12], labels[10:12]) / max_loss  # Backhand
                loss5 = loss_m(outputs[12:14], labels[12:14]) / max_loss  # BallHeight
                loss6 = loss_m(outputs[14:16], labels[14:16]) / max_loss  # Landing
                loss7 = loss_m(outputs[16:18], labels[16:18]) / max_loss  # HitterLocation
                loss8 = loss_m(outputs[18:20], labels[18:20]) / max_loss # DefenderLocation
                loss9 = loss_m(outputs[20:29], labels[20:29]) / max_loss# BallType
                loss10 = loss_m(outputs[29:32], labels[29:32]) / max_loss  # Winner

                total_loss = torch.add(
                torch.mul(0.5, loss1),
                torch.mul(0.0625, loss2),
                torch.mul(0.03125, loss3),
                torch.mul(0.03125, loss4),
                torch.mul(0.0625, loss5),
                torch.mul(0.0625, loss6),
                torch.mul(0.03125, loss7),
                torch.mul(0.03125, loss8),
                torch.mul(0.125, loss9),
                torch.mul(0.0625, loss10),)

            else:
                total_loss = 23

        return total_loss
    return badmination_loss