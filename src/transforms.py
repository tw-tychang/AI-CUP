import math, random
from typing import List, Union, Callable

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


class CustomCompose:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, img, coordXYs):
        for t in self.transforms:
            if t.__module__ != 'torchvision.transforms.transforms':
                img, coordXYs = t(img, coordXYs)
            else:
                img = t(img)
        return img, coordXYs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def RandomCrop(crop_size=(640, 640), p=0.5):
    def __RandomCrop(img: torch.Tensor, coordXYs: Union[torch.Tensor, None] = None):
        if random.random() < p:
            t, l, h, w = transforms.RandomCrop.get_params(img, crop_size)
            # imgs = [TF.crop(img, t, l, h, w) for img in imgs]
            img = TF.crop(img, t, l, h, w)

            if coordXYs is not None:
                coordXYs[0] -= l  # related with X
                coordXYs[1] -= t  # related with Y

        return img, coordXYs

    return __RandomCrop


def RandomResizedCrop(
    sizeHW: List[int] = (640, 640),
    scale: List[float] = (0.6, 1.6),
    ratio: List[float] = (3.0 / 5.0, 2.0 / 1.0),
    p: float = 0.5,
):
    def __RandomResizedCrop(img: torch.Tensor, coordXYs: Union[torch.Tensor, None] = None):
        if random.random() < p:
            t, l, h, w = transforms.RandomResizedCrop.get_params(img, scale, ratio)
            img = TF.resized_crop(img, t, l, h, w, size=sizeHW)
            if coordXYs is not None:
                coordXYs[0] *= (coordXYs[0] - l) * sizeHW[1] / w
                coordXYs[1] *= (coordXYs[1] - t) * sizeHW[0] / h
        else:
            w, h = img.shape[-1], img.shape[-2]
            img = TF.resize(img, size=sizeHW)
            if coordXYs is not None:
                coordXYs[0] *= sizeHW[1] / w
                coordXYs[1] *= sizeHW[0] / h

        return img, coordXYs

    return __RandomResizedCrop


def RandomHorizontalFlip(p=0.5):
    def __HorizontalFlip(img: torch.Tensor, coordXYs: Union[torch.Tensor, None] = None):
        if random.random() < p:
            img = TF.hflip(img)
            if coordXYs is not None:
                coordXYs[0] = img.shape[-1] - coordXYs[0]  # related with X

        return img, coordXYs

    return __HorizontalFlip


def RandomRotation(
    degrees: List[float],
    interpolation=transforms.InterpolationMode.BILINEAR,
    expand: bool = False,
    center: Union[List[int], None] = None,
    fill: Union[List[int], None] = None,
    p=0.5,
):
    def __RandomRotation(img: torch.Tensor, coordXYs: Union[torch.Tensor, None] = None):
        if random.random() < p:
            angle = transforms.RandomRotation.get_params(degrees)
            img = TF.rotate(img, angle, interpolation, expand, center, fill)
            if coordXYs is not None:
                centerX = img.shape[-1] // 2
                centerY = img.shape[-2] // 2
                coordX = coordXYs[0] - centerX
                coordY = coordXYs[1] - centerY

                angle = angle * math.pi
                cos = math.cos(angle)
                sin = math.sin(angle)
                coordXYs[0] = coordX * cos - coordY * sin + centerX
                coordXYs[1] = coordX * sin + coordY * cos + centerY

        return img, coordXYs

    return __RandomRotation
