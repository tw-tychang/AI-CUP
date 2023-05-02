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
        cv2.imshow("background.jpg", np.array(median, dtype=np.uint8))

        cap = cv2.VideoCapture(filename)
        for f_id in range(frame_count):
            ret, frame = cap.read()
            foreground_1 = np.abs(frame - median)
            channel = frame.shape[-1]
            cv2.imshow("foreground_1.jpg", np.array(foreground_1, dtype=np.uint8))

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
            cv2.imshow("foreground_2.jpg", np.array(foreground_2, dtype=np.uint8))
            
            ##使用gussian blur
            foreground_2_gray = cv2.cvtColor(np.uint8(foreground_2) , cv2.COLOR_BGR2GRAY)
            foreground_2_blur = cv2.GaussianBlur(foreground_2_gray, ksize = (5,5), sigmaX= 0,sigmaY=0)
            cv2.imshow("foreground_blur.jpg", np.array(foreground_2_blur, dtype=np.uint8))

            #進行二值化
            b = foreground_2_blur.copy()
            b[b<=160] = 0
            b[b>160] = 255
            foreground_3 = b
            cv2.imshow("foreground_3.jpg", np.array(foreground_3, dtype=np.uint8))

            # ##找邊框
            # contours, hierarchy = cv2.findContours(foreground_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # areas = []
            # for contour in contours:
            #     area = cv2.contourArea(contour)
            #     areas.append(area)

            # #找出面積大小
            # sorted_areas = sorted(areas, reverse=True)
            # if len(sorted_areas) > 1:
            #     first_max_area_idx = areas.index(sorted_areas[0])
            #     second_max_area_idx = areas.index(sorted_areas[1])
            #     max_contour = contours[first_max_area_idx]
            #     second_max_contour = contours[second_max_area_idx]

            #     # 將前兩大面積的輪廓顯示在另一張圖像上
            #     foreground_4 = frame.copy()
            #     cv2.drawContours(foreground_4, [max_contour], -1, (0, 255, 0), 3)
            #     cv2.drawContours(foreground_4, [second_max_contour], -1, (0, 0, 255), 3)

            #     #上面白色面積,高(720)的一半360,以上的面積設為[0,1]
            #     height, width, _ = frame.shape
            #     half_height = int(height/2)
            #     x, y, w, h = cv2.boundingRect(max_contour)
            #     if y + h/2 > half_height:
            #         first_max_area_array = [1,0]
            #     else:
            #         first_max_area_array = first_max_area_idx 

            #     #下面白色面積,高(720)的一半360,,以下設為[1,0]
            #     x_2, y_2, w_2, h_2 = cv2.boundingRect(second_max_contour)
            #     if y_2  < half_height:
            #         second_max_area_array = [0,1]
            #     else:
            #         second_max_area_array = second_max_area_idx

            # print('first',first_max_area_array)
            # print('second',second_max_area_array)

            contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 計算每個輪廓的面積
            areas = [cv2.contourArea(cnt) for cnt in contours]

            # 找出最大的前兩個輪廓的索引
            max_indices = np.argsort(areas)[-2:]

            # 將前兩大面積的輪廓顯示在另一張影像中
            contour_img = np.zeros_like(foreground_3)
            cv2.drawContours(contour_img, contours[max_indices], -1, 255, -1)
            cv2.imshow('contours', np.array(contour_img, dtype=np.uint8))

            # 將影像從中間切一刀，切點為高360
            height, width = frame.shape[:2]
            split_point = (0, 360)

            # 如果最大跟第二大面積連在一起，則大於面積7000的地方，直接在高360的地方，切一刀分成兩個面積
            if hierarchy[0][max_indices[0]][3] == max_indices[1]:
                for i, cnt in enumerate(contours):
                    if hierarchy[0][i][3] == max_indices[0] and areas[i] > 7000:
                        if cv2.pointPolygonTest(cnt, split_point, False) == 1:
                            cv2.line(frame, (0, 360), (width, 360), (0, 0, 255), 2)
                            cv2.imshow('frame', frame)
                            break
                else:
                    # 最大跟最小面積不連在一起，直接分成兩個面積
                    top_contours = [contours[i] for i in range(len(contours)) if cv2.pointPolygonTest(contours[i], split_point, False) == 1]
                    bottom_contours = [contours[i] for i in range(len(contours)) if cv2.pointPolygonTest(contours[i], split_point, False) == -1]

                    # 將面積分別存儲到陣列中
                    areas_top = [cv2.contourArea(cnt) for cnt in top_contours]
                    areas_bottom = [cv2.contourArea(cnt) for cnt in bottom_contours]

                    # 找出高360以上的最大面積
                    max_index_top = np.argmax(areas_top)
                    array_01 = [max_indices[0], max_indices[1]]
                    array_10 = [max_index_top, len(top_contours) - 1]

                    # 將分割線顯示在影像上
                    cv2.line(frame, (0, 360), (width, 360), (0, 0, 255), 2)
                    cv2.drawContours(frame, top_contours, max_index_top, (0, 255, 0), 2)
                    cv2.drawContours(frame, bottom_contours, len(bottom_contours) - 1, (0, 255, 0), 2)
                    cv2.imshow('frame', frame)

            cv2.imshow("foreground_4.jpg", np.array(foreground_4, dtype=np.uint8))

            cv2.waitKey(1)
