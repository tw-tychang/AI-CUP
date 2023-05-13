import os
import argparse
import random
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import EfficientNet
import warnings
from Loss import CustomLoss
from time import sleep
from tqdm import tqdm,trange
from src.data_process import DatasetInfo,get_dataloader,Processing,get_test_dataloader
from src.transforms import CustomCompose,RandomCrop,RandomResizedCrop,RandomHorizontalFlip,RandomRotation
from lib.FileTools.FileSearcher import get_filenames
from net import BadminationNet

warnings.filterwarnings("ignore",category=UserWarning)
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    filenames = get_filenames(dir_path="/home/ting-yu/AI-CUP/Data/part1/train", specific_name="*.mp4")

    # batch_size = args.b
    # num_classes = args.num_classes  

    #argumentation
    sizeHW = (640, 640)
    argumentation_order_ls = [
        RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur([3, 3]),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, hue=0.2, contrast=0.5, saturation=0.2)], p=0.75),
        transforms.RandomPosterize(6, p=0.15),
        transforms.RandomEqualize(p=0.15),
        transforms.RandomSolarize(128, p=0.1),
        transforms.RandomInvert(p=0.05),
        transforms.RandomApply(
            [transforms.ElasticTransform(alpha=random.random() * 200.0, sigma=8.0 + random.random() * 7.0)], p=0.75
        ),
        RandomRotation(degrees=[-5, 5], p=0.75),
        RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
    ]
    train_compose = CustomCompose(
    [
        *argumentation_order_ls,
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
    ])
    test_compose = CustomCompose(
        [
            transforms.GaussianBlur([3, 3]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ]
    )

    #dataloader
    train_set, val_set, train_loader, val_loader = get_dataloader(
        str(DatasetInfo.data_dir), Processing(train_compose), dataset_rate=0.8, batch_size=8, num_workers=16)

    test_set,loader = get_test_dataloader(
    str(DatasetInfo.data_dir), Processing(test_compose), batch_size=8, num_workers=16)

    model = BadminationNet(in_seq=5,output_classes=32)
    # model = models.efficientnet_v2_m(pretrained = True)
    # model = models.resnet101(pretrained = True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.w_d)
    train_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    total_epoch = args.epochs
    print_per_iteration = 100
    save_path = './model/'
    weight_path = ''

    train_epoch_loss = []
    val_epoch_loss = []
    loss_func = CustomLoss()
    for epoch in range(total_epoch):       #loop over the dataset multiple times
        print('epoch = ',epoch)

        #train
        train_step_loss = []
        model.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            train_loss = loss_func(outputs,labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_step_loss.append(train_loss.item())

            if (i+1) % print_per_iteration == 0:
                print(f'[ep {epoch + 1}][{i + 1:5d}/{len(train_loader):5d}] loss: {train_loss.item():.3f}')
        train_epoch_loss.append(np.array(train_step_loss).mean())

        train_scheduler.step()

        #validation
        model.eval()
        with torch.no_grad():
            for data in enumerate(tqdm(val_loader)):
                val_step_loss=[]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                val_loss = loss_func(outputs,labels)
                val_step_loss.append(val_loss.item())

                if (i+1) % print_per_iteration == 0:
                    print('Validation Loss',val_loss)
            val_epoch_loss.append(np.array(val_step_loss).mean())
        
        correct = 0
        total = 0
        acc = 0
        #test
        if (epoch+1) % total_epoch ==0:
            with torch.no_grad():
                for data in tqdm(loader):
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    # calculate outputs by running images through the network
                    outputs = model(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')
            # print(type(correct),'  ',type(total))
            acc =( correct / total)
            weight_path = save_path+'epochs_'+str(epoch+1)+'_acc_'+str(acc)+'.pth'
            print('weight_path = ',weight_path)

        # if (h_acc< acc) &(acc>0.65):
        #     h_acc = acc
        #     torch.save(model, weight_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-b', type=int, default=1024, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.004, help='initial learning rate')
    # parser.add_argument('-num_classes',type=int, default=32, help='classification number')
    parser.add_argument('-w_d',type = int ,default=4e-5,help='weight_decay')
    parser.add_argument('-epochs',type = int ,default=10)  
    parser.add_argument('-gamma', type=float, default=0.98, help='Initial gamma for scheduler and the default is 0.8.')
    parser.add_argument('-step', type=int, default=4, help='Initial step for scheduler and the default is 10.')  
    parser.add_argument('-m', type=float, default=0.9, help='momentum') 
    args = parser.parse_args()
    main(args)