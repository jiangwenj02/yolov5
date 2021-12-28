import pandas as pd
from glob import glob
import os.path as osp
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='empty videos')
    parser.add_argument('--dirs', type=str, default='/data3/zzhang/tmp/test-video-1226/', help='model.pt path(s)')
    parser.add_argument('--hz', type=str, default='avi', help='video last')
    parser.add_argument('--out', type=str, default='anno1228.csv', help='video last')
    opt = parser.parse_args()

    videos = glob(osp.join(opt.dirs, '*.' + opt.hz))
    frame = pd.DataFrame(columns=['source'])
    frame['video_name'] = videos
    frame.to_csv(opt.out, index=False)
