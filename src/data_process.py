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


PROJECT_DIR = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from lib.FileTools.PickleOperator import load_pickle
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
        self.image_dir = self.dir / DatasetInfo.image_dir
        self.mask_dir = self.dir / DatasetInfo.mask_dir
        self.predict_dir = self.dir / DatasetInfo.predict_dir
        self.predict_merge_dir = self.dir / DatasetInfo.predict_merge_dir
        self.ball_mask5_dir = self.dir / DatasetInfo.ball_mask5_dir
        self.frame5_start_ls: List[str] = [
            int(f.split('.pickle')[0]) for f in get_filenames(self.ball_mask5_dir, '*.pickle', withDirPath=False)
        ]

        if isTrain:
            self.label_csv = DatasetInfo.label_dir / data_id / f'{data_id}_S2.csv'


def order_data(dataset_infos: List[DatasetInfo]):
    data_order_ls = []
    data_id2startIdx_arr = np.zeros((len(dataset_ids), 2), dtype=np.uint16)
    continue_idx = 0
    for i, dataset_info in enumerate(dataset_infos):
        data_id2startIdx_arr[i] = dataset_info.id, continue_idx
        data_order_ls.extend(dataset_info.frame5_start_ls)
        continue_idx += len(dataset_info.frame5_start_ls)

    data_order_arr = np.array(data_order_ls, dtype=np.uint16)

    return data_id2startIdx_arr, data_order_arr


def transform(imgs: List[np.ndarray], label_csv: Union[pd.Series, int] = None):
    ...

    process_imgs = imgs
    process_label = label

    imgs_arr = np.concatenate(process_imgs, axis=0)
    return imgs_arr, label


class Img5Dataset(Dataset):
    def __init__(self, dataset_ids: List[str], transform_func: transform = transform, isTrain=True) -> None:
        super(Img5Dataset, self).__init__()

        self.transform_func = transform_func
        self.isTrain = isTrain

        dataset_infos = [DatasetInfo(dataset_id, self.isTrain) for dataset_id in dataset_ids]
        self.frameID2startIdx_arr, self.data_order_arr = order_data(dataset_infos)

        if self.isTrain:
            self.label_pds = [pd.read_csv(dataset_info.label_csv) for dataset_info in dataset_infos]

    def __getitem__(self, idx) -> Any:
        frame5_start = self.data_order_arr[idx]
        data_id = np.where(self.frameID2startIdx_arr[:, 1] > idx)[0] - 1  #! error will pop up at last idx
        data_dir = DatasetInfo.dir / f'{self.frameID2startIdx_arr[data_id][0]:05d}' / DatasetInfo.predict_merge_dir

        filenames = [str(data_dir / f'{i}.jpg') for i in range(frame5_start, frame5_start + 5)]
        imgs = [cv2.imread(filename) for filename in filenames]

        if self.isTrain:
            return self.transform_func(imgs, self.label_pds)

        return super().__getitem__(idx)


if __name__ == '__main__':
    dataset_ids: List[str] = os.listdir(DataLoader.dir)
