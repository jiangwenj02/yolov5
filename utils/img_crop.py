import cv2
import os
import datetime
import numpy as np
#from util import *
from tqdm import tqdm
from collections import Counter
from PIL import Image


def crop_img(img):
    if isinstance(img,str):
        img = Image.open(img)
    arr = np.asarray(img)
    combined_arr = arr.sum(axis=-1) / (255 * 3)
    truth_map = np.logical_or(combined_arr < 0.07, combined_arr > 0.95)
    threshold = 0.6
    y_bands = np.sum(truth_map, axis=1) / truth_map.shape[1]
    top_crop_index = np.argmax(y_bands < threshold)
    bottom_crop_index = y_bands.shape[0] - np.argmax(y_bands[::-1] < threshold)

    truth_map = truth_map[top_crop_index:bottom_crop_index, :]

    x_bands = np.sum(truth_map, axis=0) / truth_map.shape[0]
    left_crop_index = np.argmax(x_bands < threshold)
    right_crop_index = x_bands.shape[0] - np.argmax(x_bands[::-1] < threshold)

    cropped_arr = arr[top_crop_index:bottom_crop_index, left_crop_index:right_crop_index, :]
    toolbar_end = cropped_arr.shape[0]
    for i in range(cropped_arr.shape[0] - 1, 0, -1):
        c = Counter([tuple(l) for l in cropped_arr[i, :, :].tolist()])
        ratio = c.most_common(1)[0][-1] / cropped_arr.shape[1]
        if ratio < 0.3:
            toolbar_end = i
            break

    cropped_arr = cropped_arr[:toolbar_end, :, :]
    return cropped_arr
    #return Image.fromarray(cropped_arr)


def change_size(read_file):
    image = cv2.imread(read_file, 1)  # 读取图片 image_name应该是变量
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape

    b_left = 0
    b_right = cols
    for i in range(cols):
        c_pixel = image[:, i]
        if np.mean(c_pixel) < 20:
            if i < cols / 2:
                b_left = i
            else:
                if i < cols:
                    b_right = i
                    break

    b_top = 0
    b_bot = rows

    for i in range(rows):
        c_pixel = image[i,:]
        if np.mean(c_pixel) < 20:
            if i < rows / 2:
                b_top = i
            else:
                if i < rows:
                    b_bot = i
                    break


    pre1_picture = image[b_top:b_bot,b_left:b_right]  # 图片截取

    return pre1_picture  # 返回图片数据

'''
source_path = "gastric_data_5cls"
# source_path = "crop_test"
# 图片来源路径
  # 图片修改后的保存路径


# if not os.path.exists(save_path):
#     os.mkdir(save_path)

file_names = file_parse(source_path)

starttime = datetime.datetime.now()
for i in tqdm(file_names):
    x = crop_img(i)
    x.save(i)
print("裁剪完毕")
'''