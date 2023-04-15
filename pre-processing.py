import os
import glob
import cv2
import pandas as pd
import numpy as np
from lib.FileTools.FileSearcher import get_filenames


def statistics(
    background: np.ndarray,
    masks: np.ndarray,
    videoPath: str,
):
    cap = cv2.VideoCapture(videoPath)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("input frame resolution : width=%d, height=%d, fps=%d, the total frame=%d" %(w,h,fps,frame_count))
    imgs = []

    if not cap.isOpened():
        print("Can not recieve frame")

    timeF = 1000
    f=0
    while ret:
        ret, frame = cap.read()
        if(f % timeF == 0):
            img_list.append(frame)
        f= f+1
        cv2.waitkey(1)
    print(img_list)

    img_array = np.array(imgs,dype=np.uint8)
    print(img_array)
    cap.release()



if __name__ == '__main__':
    get_filenames(dir_path: str, specific_name: str, withDirPath: bool=True)
