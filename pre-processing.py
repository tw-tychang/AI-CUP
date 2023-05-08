import os
import glob
from pathlib import Path
from pickletools import float8
import cv2
import pandas as pd
import numpy as np
from lib.FileTools.FileSearcher import get_filenames


if __name__ == "__main__":
    filenames = get_filenames(
        dir_path="/home/ting-yu/AI-CUP/Data/part1/train", specific_name="*.mp4"
    )
    # print(filenames)

    img_list = []
    timeF = 20
    f = 0
    videos = []
    for filename in filenames:
        cap = cv2.VideoCapture(filename)
        # videos.append(cap)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print("input frame resolution : width=%d, height=%d, fps=%d, the total frame=%d" %(w,h,fps,frame_count))

        img_list = []
        for f_id in range(frame_count):
            frame: np.ndarray
            ret, frame = cap.read()

            if ret is False:
                continue
            if frame_count > 1000:
                if f_id % (fps // 2) == 0:
                    img_list.append(frame)

            elif f_id % (fps * 0.1) == 0:
                img_list.append(frame)

        aa = 0

        imgs_sample = np.array(img_list, dtype=np.uint8)
        cap.release()

        std: np.ndarray = np.std(imgs_sample, axis=0)
        mean: np.ndarray = np.mean(imgs_sample, axis=0)
        median: np.ndarray = np.median(imgs_sample, axis=0)
        print(f"{Path(filename).parent}")
        cv2.imwrite(
            f"{Path(filename).parent}/background.jpg", np.array(median, dtype=np.uint8)
        )
        # cv2.imshow("background.jpg", np.array(median, dtype=np.uint8))

        cap = cv2.VideoCapture(filename)
        for f_id in range(frame_count):
            ret, frame = cap.read()
            foreground_1 = np.abs(frame - median)
            channel = frame.shape[-1]
            # cv2.imshow("foreground_1.jpg", np.array(foreground_1, dtype=np.uint8))

            useid = None
            a = foreground_1.copy()
            grayid = np.where(a < (127, 127, 127))[:-1]
            bgr_max = np.amax(a[grayid], axis=1)
            bgr_min = np.amin(a[grayid], axis=1)
            diff: np.ndarray = bgr_max - bgr_min

            # hist, bins = np.histogram(diff[diff > 0], 25, [0, 255])
            # for h, b in zip(hist, bins[1:]):
            #     print(f"{int(b)}: {h}")

            # aa = diff[diff < 20]
            # print(aa[aa > 0].shape[0] / diff[diff > 0].shape[0] - 0.5)
            # print(diff[diff < 20].shape[0] / diff.size - 0.5)
            # print()

            diff_mean = np.mean(diff)
            diff_std = np.std(diff)
            diff_mean_right_tail = 1.645 * diff_std + diff_mean   # z-score 0.95

            useid = np.where(diff < int(diff_mean_right_tail))
            cc = grayid[0][useid[0]], grayid[1][useid[0]]
            a[cc] = (0, 0, 0)
            foreground_2 = a
            foreground_2[foreground_2 > 0] = 255
            # cv2.imshow("foreground_2.jpg", np.array(foreground_2, dtype=np.uint8))
            
            ##使用gussian blur
            foreground_2_gray = cv2.cvtColor(np.uint8(foreground_2) , cv2.COLOR_BGR2GRAY)
            foreground_2_blur = cv2.GaussianBlur(foreground_2_gray, ksize = (5,5), sigmaX= 0,sigmaY=0)
            kernel_3 = np.ones((3,3),np.uint8)
            kernel_5 = np.ones((5,5),np.uint8)
            foreground_2_blur = cv2.morphologyEx(foreground_2_blur,cv2.MORPH_ERODE,kernel_3)
            foreground_2_blur = cv2.morphologyEx(foreground_2_blur,cv2.MORPH_DILATE,kernel_5)
            # foreground_2_blur = cv2.morphologyEx(foreground_2_blur,cv2.MORPH_CLOSE,kernel_5)
            cv2.imshow("foreground_blur.jpg", np.array(foreground_2_blur, dtype=np.uint8))

            #進行二值化
            b = foreground_2_blur.copy()
            b[b<=160] = 0
            b[b>160] = 255
            foreground_3 = b
            # cv2.imshow("foreground_3.jpg", np.array(foreground_3, dtype=np.uint8))

            # frame_gray = cv2.cvtColor(np.uint8(frame) , cv2.COLOR_BGR2GRAY)
            #把一些會影響contour結果的地方蓋掉
            mask_1 = foreground_3.copy()
            mask_1[0:105, 0:455] = 0
            mask_1[0:72, 0:1280] = 0
            mask_1[648:720, 0:1280] = 0
            mask_1[0:720, 0:192] = 0
            mask_1[0:720, 1088:1280] = 0
            mask_1[0:440, 1000:1280] = 0
            mask_1[0:383, 975:1000] = 0
            mask_1[0:320, 944:975] = 0
            mask_1[0:269, 918:944] = 0
            cv2.imshow("mask_b.jpg", np.array(mask_1, dtype=np.uint8))

            contours, hierarchy = cv2.findContours(mask_1 ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            areas = [cv2.contourArea(c) for c in contours]
    
            # 找到面積最大的前兩個輪廓
            sorted_areas = sorted(areas, reverse=True)[:2]
            max_contours = [contours[areas.index(a)] for a in sorted_areas]

            # 如果有兩個及以上的輪廓
            if len(sorted_areas) >= 2:
                max_area_idx1 = areas.index(sorted_areas[0])
                max_area_idx2 = areas.index(sorted_areas[1])
                max_contour1 = contours[max_area_idx1]
                max_contour2 = contours[max_area_idx2]
                max_area1 = sorted_areas[0]
                max_area2 = sorted_areas[1]
                print('max',max_area1)
                print('second',max_area2)

                # 計算每個輪廓的重心
                moments = [cv2.moments(cnt) for cnt in contours]

                # 將高度大於360的輪廓面積設為 [0, 1]
                # 將高度小於等於360的輪廓面積設為 [1, 0]
                height, width = frame.shape[:2]
                areas_above = []
                areas_below = []
                mask_2 = mask_1.copy()
                for i, cnt in enumerate(contours):
                    if moments[i]['m00'] > 0:
                        cx = int(moments[i]['m10']/moments[i]['m00'])
                        cy = int(moments[i]['m01']/moments[i]['m00'])
                        # x, y, w, h = cv2.boundingRect(max_contours)
                        if max_area2 <=1100 and max_area1 >= 4000:         #如果在這個區間，只顯現最大的面積
                            contours_up = mask_2[:cy]
                            contours_up[contours_up == 255] = 1
                            contours_down = mask_2[360:720]
                            contours_down[contours_down == 255] = 2
                            cv2.drawContours(frame, [max_contours[0]], -1, (0, 255, 0), 3)
                            cv2.imshow('Contours', frame)
                            cv2.drawContours(mask_2,contours, max_area_idx1, 255, cv2.FILLED)
                            cv2.imshow("mask_2.jpg", np.array(mask_2, dtype=np.uint8))
                        else:                                              #如果在這個區間，顯現最大與第二大的面積
                            cv2.drawContours(frame, max_contours, -1, (0, 255, 0), 3)
                            cv2.imshow('Contours', frame)
                            cv2.drawContours(mask_2,max_contours, -1, 255, cv2.FILLED)
                            cv2.imshow("mask_2.jpg", np.array(mask_2, dtype=np.uint8))
                            if cy > height/2:
                                contours_up = mask_2[:360]
                                contours_up[contours_up == 255] = 1
                            elif cy < height/2:
                                contours_down = mask_2[360:720]
                                contours_down[contours_down == 255] = 2

                print('up',contours_up)
                print('down',contours_down)

            cv2.waitKey(1)
