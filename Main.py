import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import warnings
from data_parallel_my_v2 import BalancedDataParallel
from Loss import CustomLoss
from tqdm import tqdm
from src.data_process import DatasetInfo, get_dataloader, Processing, get_test_dataloader
from src.transforms import CustomCompose, IterativeCustomCompose, RandomCrop, RandomResizedCrop, RandomHorizontalFlip, RandomRotation
from lib.FileTools.FileSearcher import get_filenames
from lib.ML_Tools.ModelPerform import ModelPerform
from net import BadminationNet


warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # batch_size = args.b

    # argumentation
    sizeHW = (512, 512)
    argumentation_order_ls = [
        RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
        RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur([3, 3]),
        # transforms.RandomApply([transforms.ColorJitter(brightness=0.4, hue=0.2, contrast=0.5, saturation=0.2)], p=0.75),
        # transforms.RandomPosterize(6, p=0.15),
        # transforms.RandomEqualize(p=0.15),
        # transforms.RandomSolarize(128, p=0.1),
        # transforms.RandomInvert(p=0.05),
        transforms.RandomApply(
            [transforms.ElasticTransform(alpha=random.random() * 200.0, sigma=8.0 + random.random() * 7.0)], p=0.25
        ),
        RandomRotation(degrees=[-5, 5], p=0.75),
    ]
    train_compose = CustomCompose(
        [
            *argumentation_order_ls,
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ]
    )
    train_iter_compose = IterativeCustomCompose(
        [
            *argumentation_order_ls,
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ],
        transform_img_size=sizeHW,
    )
    test_iter_compose = IterativeCustomCompose(
        [
            RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ],
        transform_img_size=sizeHW,
    )
    test_compose = CustomCompose(
        [
            transforms.GaussianBlur([3, 3]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ]
    )

    # dataloader
    train_loader, val_loader = get_dataloader(
        filenames=[
            *get_filenames('Data/dataset_pickle/hit', specific_name="*.pickle"),
            *get_filenames('Data/dataset_pickle/miss', specific_name="*.pickle")[:6000],
        ],
        dataset_rate=0.8,
        batch_size=23,
        num_workers=46,
        pin_memory=True,
        usePickle=True,
        compose=train_compose,
    )[2:]

    # filenames = [
    #     *get_filenames('Data/dataset_pickle/hit', specific_name="*.pickle"),
    #     *get_filenames('Data/dataset_pickle/miss', specific_name="*.pickle")[:4000],
    # ]
    # random.shuffle(filenames)
    # filenames = filenames[:100]

    # # dataloader
    # train_loader, val_loader = get_dataloader(
    #     filenames=filenames,
    #     dataset_rate=0.8,
    #     batch_size=23,
    #     num_workers=46,
    #     pin_memory=True,
    #     usePickle=True,
    #     compose=train_compose,
    # )[2:]

    # test_set, loader = get_test_dataloader(str(DatasetInfo.data_dir), Processing(test_compose), batch_size=65, num_workers=46)

    gpu0_bsz = 7
    acc_grad = 1
    # File = '/root/Work/AI-CUP/out/loss937.04813.pt'
    bad_model = BadminationNet(in_seq=5)

    model = BalancedDataParallel(gpu0_bsz, bad_model, dim=0).to(device)
    model.update = bad_model.update

    # model.load_state_dict(torch.load(File))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.w_d, momentum=args.m)
    train_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    total_epoch = args.epochs
    print_per_iteration = 100

    loss_func = CustomLoss().to(device)
    model_perform = ModelPerform(
        train_loss_ls=[], train_acc_ls=[], test_loss_ls=[], test_acc_ls=[]
    )  # ,train_acc_ls=[],test_acc_ls=[])

    total_epoch += model_perform.num_epoch
    for model_perform.num_epoch in range(model_perform.num_epoch + 1, total_epoch + 1):  # loop over the dataset multiple times
        print('epoch = ', model_perform.num_epoch)

        #!train
        best_val_loss = 999999.0
        train_sum_loss = 0.0
        model.train()
        inputs: torch.Tensor
        labels: torch.Tensor
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                batch_coordXYs = torch.stack(
                    [labels[:, 14:20:2], labels[:, 15:20:2]],
                ).permute(
                    1, 0, 2
                )  # stack like: [[relatedX, ...], [relatedY, ...]]

                inputs, batch_coordXYs = train_iter_compose(inputs, batch_coordXYs)
                batch_coordXYs = batch_coordXYs.permute(1, 0, 2)
                labels[:, 14:20:2] = batch_coordXYs[0]
                labels[:, 15:20:2] = batch_coordXYs[1]

            outputs = model(inputs)
            model.update(outputs, labels)

            # todo: need to add the general loss
            train_sum_loss += loss_func(outputs, labels).item()
        model_perform.train_loss_ls.append(train_sum_loss / len(train_loader))
        model_perform.train_acc_ls.append(0.0)

        if (i + 1) % print_per_iteration == 0:
            print(f'[{model_perform.num_epoch + 1:3}/{total_epoch}] loss: {model_perform.train_loss_ls[-1]:.5f}')

        # train_scheduler.step()

        # validation
        sum_loss = 0
        err_num = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    batch_coordXYs = torch.stack(
                        [labels[:, 14:20:2], labels[:, 15:20:2]]
                    )  # stack like: [[relatedX, ...], [relatedY, ...]]

                    inputs, batch_coordXYs = test_iter_compose(inputs, batch_coordXYs)
                    labels[:, 14:20:2] = batch_coordXYs[0]
                    labels[:, 15:20:2] = batch_coordXYs[1]

                try:
                    outputs = model(inputs)
                except ValueError:
                    err_num += 1
                    continue

                # print(outputs)
                val_loss = loss_func(outputs, labels)
                sum_loss += val_loss.item()

        model_perform.test_loss_ls.append(sum_loss / (len(val_loader) - err_num))
        model_perform.test_acc_ls.append(0.0)

        if best_val_loss > model_perform.test_loss_ls[-1]:
            best_val_loss = model_perform.test_loss_ls[-1]
            torch.save(model.state_dict(), f"out/best_loss{best_val_loss:.5f}.pt")
        elif model_perform.num_epoch % 5 == 0:
            torch.save(model.state_dict(), f"out/loss{model_perform.test_loss_ls[-1]:.5f}.pt")

        model_perform.save_history_csv()
        model_perform.draw_plot(0, model_perform.num_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-b', type=int, default=1024, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    # parser.add_argument('-num_classes',type=int, default=32, help='classification number')
    parser.add_argument('-w_d', type=int, default=4e-5, help='weight_decay')
    parser.add_argument('-epochs', type=int, default=25)
    parser.add_argument('-gamma', type=float, default=0.98, help='Initial gamma for scheduler and the default is 0.8.')
    parser.add_argument('-step', type=int, default=4, help='Initial step for scheduler and the default is 10.')
    parser.add_argument('-m', type=float, default=0.9, help='momentum')
    args = parser.parse_args()
    main(args)
