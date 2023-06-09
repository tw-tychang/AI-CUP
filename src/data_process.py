import os
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from lib.FileTools.PickleOperator import load_pickle
from lib.FileTools.WordOperator import str_format
from lib.FileTools.FileSearcher import get_filenames
from src.transforms import CustomCompose, RandomHorizontalFlip, RandomResizedCrop, RandomRotation


class DatasetInfo:
    data_dir = Path('./Data/result')
    label_dir = Path('./Data/part1/train')
    image_dir = Path('image')
    predict_dir = Path('predict')
    mask_dir = Path('mask')
    predict_merge_dir = Path('predict_merge')
    ball_mask5_dir = Path('ball_mask5_dir')

    def __init__(self, data_id: str, isTrain=True) -> None:
        self.id = int(data_id)
        self.data_dir = DatasetInfo.data_dir / data_id
        self.image_dir = self.data_dir / DatasetInfo.image_dir
        self.mask_dir = self.data_dir / DatasetInfo.mask_dir
        self.predict_dir = self.data_dir / DatasetInfo.predict_dir
        self.predict_merge_dir = self.data_dir / DatasetInfo.predict_merge_dir
        self.ball_mask5_dir = self.data_dir / DatasetInfo.ball_mask5_dir
        self.frame5_start_ls: List[str] = [
            int(f.split('.pickle')[0]) for f in get_filenames(str(self.ball_mask5_dir), '*.pickle', withDirPath=False)
        ]

        if isTrain:
            self.label_csv = DatasetInfo.label_dir / data_id / f'{data_id}_S2.csv'


class CSVColumnNames:
    ShotSeq = 'ShotSeq'
    HitFrame = 'HitFrame'
    Hitter = 'Hitter'
    RoundHead = 'RoundHead'
    Backhand = 'Backhand'
    BallHeight = 'BallHeight'
    LandingX = 'LandingX'
    LandingY = 'LandingY'
    HitterLocationX = 'HitterLocationX'
    HitterLocationY = 'HitterLocationY'
    DefenderLocationX = 'DefenderLocationX'
    DefenderLocationY = 'DefenderLocationY'
    BallType = 'BallType'
    Winner = 'Winner'


def order_data(dataset_infos: List[DatasetInfo], len_dataset: int):
    data_order_ls = []
    data_id2startIdx_arr = np.zeros((len_dataset, 2), dtype=np.uint32)
    continue_idx = 0
    for i, dataset_info in enumerate(dataset_infos):
        data_id2startIdx_arr[i] = dataset_info.id, continue_idx
        data_order_ls.extend(dataset_info.frame5_start_ls)
        continue_idx += len(dataset_info.frame5_start_ls)

    data_order_arr = np.array(data_order_ls, dtype=np.uint32)

    return data_id2startIdx_arr, data_order_arr


def Processing(compose: Union[CustomCompose, transforms.Compose]):
    def __processing(imgs: List[np.ndarray], label_info: Union[pd.Series, int] = 0):
        imgs = [TF.to_pil_image(img) for img in imgs]
        process_imgs = torch.stack([TF.to_tensor(img) for img in imgs]).type(torch.uint8)

        if not isinstance(label_info, int):  # hit_frame in it
            process_label = torch.zeros(32, dtype=torch.float32)
            process_label[label_info.at[CSVColumnNames.HitFrame]] = 1.0  # HitFrame: [0~5] one-hot
            process_label[6 if label_info.at[CSVColumnNames.Hitter] == 'A' else 7] = 1.0  # Hitter: [6,7] one-hot
            process_label[7 + label_info.at[CSVColumnNames.RoundHead]] = 1.0  # RoundHead: [8,9] one-hot
            process_label[9 + label_info.at[CSVColumnNames.Backhand]] = 1.0  # Backhand: [10,11] one-hot
            process_label[11 + label_info.at[CSVColumnNames.BallHeight]] = 1.0  # BallHeight: [12,13] one-hot
            process_label[14:20] = torch.from_numpy(
                label_info.loc[
                    [
                        CSVColumnNames.LandingX,  # LandingX: 14
                        CSVColumnNames.LandingY,  # LandingY: 15
                        CSVColumnNames.HitterLocationX,  # HitterLocationX: 16
                        CSVColumnNames.HitterLocationY,  # HitterLocationY: 17
                        CSVColumnNames.DefenderLocationX,  # DefenderLocationX: 18
                        CSVColumnNames.DefenderLocationY,  # DefenderLocationY: 19
                    ],
                ].to_numpy(dtype=np.float32)
            )
            process_label[19 + label_info.at[CSVColumnNames.BallType]] = 1.0  # BallType: [20~28] one-hot

            w_id = 20
            w = label_info.at[CSVColumnNames.Winner]
            if w == 'B':
                w_id += 1
            elif w == 'X':
                w_id += 2
            process_label[w_id] = 1.0  # Winner: [29~31] one-hot

            coordXYs = torch.stack([process_label[14:20:2], process_label[15:20:2]])

            process_imgs, coordXYs = compose(process_imgs, coordXYs)
            process_label[14:20:2] = coordXYs[0]
            process_label[15:20:2] = coordXYs[1]

        if label_info == 0:  # test stage
            return compose(process_imgs)
        elif label_info == -1:  # hit_frame miss
            process_label = torch.zeros(6, dtype=torch.float32)
            process_label[-1] = 1.0
            process_imgs, _ = compose(process_imgs, None)
            return process_imgs, process_label

    return __processing


class Img5Dataset(Dataset):
    def __init__(self, dataset_ids: List[str], processing: Processing, isTrain=True) -> None:
        super(Img5Dataset, self).__init__()

        self.processing = processing
        self.isTrain = isTrain

        dataset_infos = [DatasetInfo(dataset_id, self.isTrain) for dataset_id in dataset_ids]
        self.frameID2startIdx_arr, self.data_order_arr = order_data(dataset_infos, len(dataset_ids))

        if self.isTrain:
            self.label_csvs = [dataset_info.label_csv for dataset_info in dataset_infos]

    def __getitem__(self, idx):
        frame5_start = self.data_order_arr[idx]
        data_id = np.where(self.frameID2startIdx_arr[:, 1] <= idx)[0][-1]
        data_dir = DatasetInfo.data_dir / f'{self.frameID2startIdx_arr[data_id][0]:05d}' / DatasetInfo.predict_merge_dir

        filenames = [str(data_dir / f'{i}.jpg') for i in range(frame5_start + 1, frame5_start + 6)]  # "*.jpg" start from 1
        imgs = [cv2.imread(filename) for filename in filenames]  # TODO: can change to use torchvision.io.read_image()
        imgs = [imgs[i - 1].copy() if img is None else img for i, img in enumerate(imgs)]

        if self.isTrain:
            df = pd.read_csv(str(self.label_csvs[data_id]))
            hit_frames = df.loc[:, CSVColumnNames.HitFrame].to_numpy()
            hit_idx = np.where((frame5_start <= hit_frames) & (hit_frames < (frame5_start + 5)))[0]
            if hit_idx.size == 0:
                return self.processing(imgs, label_info=-1)

            hit_idx = hit_idx[0]
            df.at[hit_idx, CSVColumnNames.HitFrame] -= frame5_start
            return self.processing(imgs, label_info=df.loc[hit_idx])
        else:
            return self.processing(imgs, label_info=0)

    def __len__(self):
        return self.data_order_arr.shape[0]


def get_dataloader(
    preprocess_dir: str = str(DatasetInfo.data_dir),
    Processing=Processing,
    dataset_rate=0.8,
    batch_size: int = 32,
    num_workers: int = 8,
):
    dataset_ids: List[str] = os.listdir(preprocess_dir)
    dataset = Img5Dataset(dataset_ids, processing=Processing)

    train_len = int(len(dataset) * dataset_rate)
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_set, val_set, train_loader, val_loader


if __name__ == '__main__':
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    sizeHW = (640, 640)
    compose = CustomCompose(
        [
            RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, hue=0.2, contrast=0.5, saturation=0.2)], p=0.75),
            transforms.RandomPosterize(6, p=0.25),
            transforms.RandomEqualize(p=0.25),
            transforms.RandomSolarize(128, p=0.15),
            transforms.RandomInvert(p=0.1),
            transforms.RandomApply(
                [transforms.ElasticTransform(alpha=random.random() * 200.0, sigma=8.0 + random.random() * 7.0)], p=0.75
            ),
            RandomRotation(degrees=[-5, 5], p=0.75),
            RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
        ]
    )

    # dataset_ids: List[str] = os.listdir(str(DatasetInfo.data_dir))
    # dataset = Img5Dataset(dataset_ids, processing=Processing(compose))

    # check_start = len(dataset) - (len(dataset) // 4)
    # check_end = len(dataset)
    # error_ids = []
    # error_msgs = []
    # for i in range(check_start, check_end):
    #     try:
    #         data, label = dataset[i]
    #     except Exception as e:
    #         error_msg = f'{i}, {e}'
    #         print(str_format(error_msg, fore='r'))
    #         error_ids.append(i)
    #         error_msgs.append(error_msg)

    #     if i % 100 == 0:
    #         print(i)

    # # dataset[check_start]

    # # print(len(dataset))
    # # print(str_format(f"error_ids: {error_ids}", fore='y'))
    # # print(error_msgs)

    # train_len = int(len(dataset) * 0.8)
    # train_set, val_set = random_split(dataset, len(dataset) - train_len)

    # dataset[1]

    train_set, val_set, train_loader, val_loader = get_dataloader(
        str(DatasetInfo.data_dir), Processing(compose), dataset_rate=0.8, batch_size=32, num_workers=8
    )

    for data, label in val_loader:
        aa = 0
