import os
import os.path as osp
import imageio
import pandas as pd
import datetime
from collections import defaultdict
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import multiprocessing
import cv2
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadVideos
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import torch
from pathlib import Path
import mmcv
import time
from datetime import timedelta
from utils.img_crop import crop_img
import glob
from math import ceil

def my_reshape(arr, cols):
    rows = ceil(len(arr) / cols)
    res = []
    for row in range(rows):
        current_row = []
        for col in range(cols):
            arr_idx = row * cols + col
            if arr_idx < len(arr):
                current_row.append(arr[arr_idx])
            else:
                current_row.append(None)
        res.append(current_row)
    return res

rois = {
    'big': [441, 1, 1278, 720],  # july video
    'small': [156, 40, 698, 527],
    '20191009_0900_0915': [156, 33, 699, 439],
    '20191009_0900_0915': [156, 33, 699, 439]
}
SKIP_FRAME = 1
CONTINUE_FRAME = 1
LABEL = ['erosive','ulcer']
COLOR = [(255,0,0),(0,255,0)]

def time_in_list_range( range_list, x):
    for i, range in enumerate(range_list):
        if time_in_range(range[0], range[1], x):
            return True, i
    return False, 0

def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end



def my_reshape(arr, cols):
    rows = ceil(len(arr) / cols)
    res = []
    for row in range(rows):
        current_row = []
        for col in range(cols):
            arr_idx = row * cols + col
            if arr_idx < len(arr):
                current_row.append(arr[arr_idx])
            else:
                current_row.append(None)
        res.append(current_row)
    return res


class Evaluator:
    def __init__(self, opt):

        self.opt = opt
        self.saving_root = opt.save_path
        self.video_root = opt.video_path
        self.interval = opt.interval
        self.saving_root = osp.join(self.saving_root, osp.basename(self.video_root))
        os.makedirs(self.saving_root, exist_ok=True)
        os.popen('rm -r ' + osp.join(self.saving_root, '*'))
        self.nums = 3

    def concat_tile(self, im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

    def extract_video(self):
        im_list_2d = []
        # Read the video from specified path
        cam = cv2.VideoCapture(self.video_root)        
        try:            
            # creating a folder named data
            if not os.path.exists('data'):
                os.makedirs('data')
        
        # if not created then raise error
        except OSError:
            print ('Error: Creating directory of data')
        
        fps = cam.get(cv2.CAP_PROP_FPS)
        all_frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
        save_intervel = all_frame // (self.nums * self.nums)
        # frame
        currentframe = 0
        while(True):            
            # reading from frame
            ret,frame = cam.read()            
            if ret:
                crop = False
                if currentframe % self.interval == 0:
                    # if video is still left continue creating images
                    td = timedelta(seconds=(currentframe / fps))
                    # basename = '{}.jpg'.format(str(td))    
                    frame = crop_img(frame)
                    crop = True                
                    # frame = frame[30:267, 87:404, :]
                    basename = '{0:08d}.jpg'.format(currentframe)
                    name = osp.join(self.saving_root, basename)           
                    # savePath = (r"D:\sxl\处理图片\汉字分类\train653_badHandle\%d.jpg" % (count))         
                    # writing the extracted images
                    cv2.putText(frame, '#{0:08d}'.format(currentframe), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (124,205,124), 2)
                    # cv2.imwrite(name, frame)
                    cv2.imencode('.jpg',frame)[1].tofile(name)
                    
                    print('saving image %s' % (name))

                if currentframe % save_intervel == 0 and len(im_list_2d) < self.nums * self.nums:
                    if crop ==False:
                        cv2.putText(frame, '#{0:08d}'.format(currentframe), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (124,205,124), 2)
                        frame = crop_img(frame)
                    frame = cv2.resize(frame, (300,200), interpolation = cv2.INTER_AREA)
                    im_list_2d.append(frame)
                    # increasing counter so that it will
                    # show how many frames are created
                currentframe += 1
            else:
                break
        print(len(im_list_2d))
        im_list_2d = my_reshape(im_list_2d, self.nums)
        ims = self.concat_tile(im_list_2d)
        name = osp.join(self.saving_root, 'summary.jpg')
        cv2.imencode('.jpg',ims)[1].tofile(name)
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

        

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="video_evaluation")
    parser.add_argument('--video_path', type=str, default='E:/Users/Raytine/Documents/蜂群/video/Demo/vrvideos/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_path', type=str, default='E:/Users/Raytine/Documents/蜂群/video/Demo/vrvideos/images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--interval', type=int, default=5, help='source')  # file/folder, 0 for webcam
    
    args = parser.parse_args()
    files = glob.glob(args.video_path + '*.mp4')
    for file in files:
        args.video_path = file
        evaluator = Evaluator(args)
        evaluator.extract_video()


