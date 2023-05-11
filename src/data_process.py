import os
from pathlib import Path
from typing import List, Union

import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms.functional import to_tensor


PROJECT_DIR = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from lib.FileTools.PickleOperator import load_pickle
from lib.FileTools.WordOperator import str_format
from lib.FileTools.FileSearcher import get_filenames


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


def order_data(dataset_infos: List[DatasetInfo]):
    data_order_ls = []
    data_id2startIdx_arr = np.zeros((len(dataset_ids), 2), dtype=np.uint32)
    continue_idx = 0
    for i, dataset_info in enumerate(dataset_infos):
        data_id2startIdx_arr[i] = dataset_info.id, continue_idx
        data_order_ls.extend(dataset_info.frame5_start_ls)
        continue_idx += len(dataset_info.frame5_start_ls)

    data_order_arr = np.array(data_order_ls, dtype=np.uint32)

    return data_id2startIdx_arr, data_order_arr


def transform(imgs: List[np.ndarray], label_info: Union[pd.Series, int] = 0):
    CNs = CSVColumnNames

    process_imgs = imgs
    process_imgs = np.concatenate(process_imgs, axis=2)
    process_imgs = to_tensor(process_imgs)

    if type(label_info) is int:
        if label_info == 0:  # test stage
            return process_imgs

        elif label_info == -1:  # hit_frame miss
            process_label = torch.zeros(6, dtype=torch.float32)
            process_label[-1] = 1.0
            return process_imgs, process_label

    else:  # hit_frame in it
        process_label = torch.zeros(32, dtype=torch.float32)
        process_label[label_info.at[CNs.HitFrame]] = 1.0  # HitFrame: [0~5] one-hot
        process_label[6 if label_info.at[CNs.Hitter] == 'A' else 7] = 1.0  # Hitter: [6,7] one-hot
        process_label[7 + label_info.at[CNs.RoundHead]] = 1.0  # RoundHead: [8,9] one-hot
        process_label[9 + label_info.at[CNs.Backhand]] = 1.0  # Backhand: [10,11] one-hot
        process_label[11 + label_info.at[CNs.BallHeight]] = 1.0  # BallHeight: [12,13] one-hot

        process_label[14:20] = torch.from_numpy(
            label_info.loc[
                [
                    CNs.LandingX,  # LandingX: 14
                    CNs.LandingY,  # LandingY: 15
                    CNs.HitterLocationX,  # HitterLocationX: 16
                    CNs.HitterLocationY,  # HitterLocationY: 17
                    CNs.DefenderLocationX,  # DefenderLocationX: 18
                    CNs.DefenderLocationY,  # DefenderLocationY: 19
                ],
            ].to_numpy(dtype=np.float32)
        )
        process_label[19 + label_info.at[CNs.BallType]] = 1.0  # BallType: [20~28] one-hot

        w_id = 20
        w = label_info.at[CNs.Winner]
        if w == 'B':
            w_id += 1
        elif w == 'X':
            w_id += 2

        process_label[w_id] = 1.0  # Winner: [29~31] one-hot

        return process_imgs, process_label


class Img5Dataset(Dataset):
    def __init__(self, dataset_ids: List[str], transform_func: transform = transform, isTrain=True) -> None:
        super(Img5Dataset, self).__init__()

        self.transform_func = transform_func
        self.isTrain = isTrain

        dataset_infos = [DatasetInfo(dataset_id, self.isTrain) for dataset_id in dataset_ids]
        self.frameID2startIdx_arr, self.data_order_arr = order_data(dataset_infos)

        if self.isTrain:
            self.label_csvs = [dataset_info.label_csv for dataset_info in dataset_infos]

    def __getitem__(self, idx):
        frame5_start = self.data_order_arr[idx]
        data_id = np.where(self.frameID2startIdx_arr[:, 1] <= idx)[0][-1]
        data_dir = DatasetInfo.data_dir / f'{self.frameID2startIdx_arr[data_id][0]:05d}' / DatasetInfo.predict_merge_dir

        filenames = [str(data_dir / f'{i}.jpg') for i in range(frame5_start + 1, frame5_start + 6)]  # "*.jpg" start from 1
        imgs = [cv2.imread(filename) for filename in filenames]
        imgs = [imgs[i - 1].copy() if img is None else img for i, img in enumerate(imgs)]

        if self.isTrain:
            df = pd.read_csv(str(self.label_csvs[data_id]))
            hit_frames = df.loc[:, CSVColumnNames.HitFrame].to_numpy()
            hit_idx = np.where((frame5_start <= hit_frames) & (hit_frames < (frame5_start + 5)))[0]
            if hit_idx.size == 0:
                return self.transform_func(imgs, label_info=-1)

            hit_idx = hit_idx[0]
            df.at[hit_idx, CSVColumnNames.HitFrame] -= frame5_start
            return self.transform_func(imgs, label_info=df.loc[hit_idx])
        else:
            return self.transform_func(imgs, label_info=0)

    def __len__(self):
        return self.data_order_arr.shape[0]


if __name__ == '__main__':
    dataset_ids: List[str] = os.listdir(str(DatasetInfo.data_dir))
    dataset = Img5Dataset(dataset_ids, transform_func=transform)

    check_start = len(dataset) - (len(dataset) // 4)
    check_end = len(dataset)
    error_ids = []
    error_msgs = []
    for i in range(check_start, check_end):
        try:
            data, label = dataset[i]
        except Exception as e:
            error_msg = f'{i}, {e}'
            print(str_format(error_msg, fore='r'))
            error_ids.append(i)
            error_msgs.append(error_msg)

        if i % 100 == 0:
            print(i)

    print(len(dataset))
    print(str_format(f"error_ids: {error_ids}", fore='y'))
    print(error_msgs)

    # dataset[1]
