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

class Evaluator:
    def __init__(self, opt):

        self.opt = opt
        self.saving_root = opt.save_path
        self.video_root = opt.video_path
        self.interval = opt.interval
        os.popen('rm -r ' + self.saving_root + '*')

    def extract_video(self):
  
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
        # frame
        currentframe = 0
        
        while(True):            
            # reading from frame
            ret,frame = cam.read()            
            if ret:
                if frame % self.interval == 0:
                    # if video is still left continue creating images
                    td = timedelta(seconds=(frame / fps))
                    name = osp.join(self.saving_root, str(td) + '.jpg')
                    frame = crop_img(frame)
                    # writing the extracted images
                    cv2.imwrite(name, frame)

                    print('saving image %s' % (name))
            
                    # increasing counter so that it will
                    # show how many frames are created
                    currentframe += 1
            else:
                break
  
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

        

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="video_evaluation")
    parser.add_argument('--video_path', type=str, default='/data2/qilei_chen/DATA/erosive_ulcer_videos', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_path', type=str, default='/data3/zzhang/tmp/erosive_ulcer_videos0615/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--interval', type=int, default=5, help='source')  # file/folder, 0 for webcam
    
    args = parser.parse_args()
    evaluator = Evaluator(args)

    evaluator.extract_video()


