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
import pandas as pd
from pathlib import Path
import mmcv
import time

class Evaluator:
    def __init__(self, opt):

        self.opt = opt
        self.saving_root = opt.save_path
        summary_file = osp.join(self.saving_root, 'summary.txt')
        self.save_file = osp.join(self.saving_root, 'output.csv')
        self.det_summary = parser_summary()
        self.det_summary.open_summary(summary_file)
        os.makedirs(self.saving_root, exist_ok=True)

    def plot_tables(self, gts):
        table=pd.DataFrame()
        for temp in gts:
            video, anno, break_time, break_time_name = temp['video_name'], temp['tp_range'], temp['time_break'], temp['time_break_name']
            if video.replace('.avi', '') not in self.det_summary.summary:
                print(video)
                continue
            det_result = self.det_summary.summary[video.replace('.avi', '')]
            print(det_result)
            first_frame = det_result['start'][0]
            det_frames_in = [int(float(frames) * float(ratios) / 100) for frames, ratios in zip(det_result['all_fps'][1:-1:2], det_result['det_Pro'][1:-1:2])]
            frames_in = [int(frames) for frames in det_result['all_fps'][1:-1:2]]
            det_frames_out = [int(frames) for frames in det_result['all_fps'][0:-1:2]]
            det_frame_ratio_in = sum(det_frames_in) / (sum(frames_in) + 1)
            det_frames_out = sum(det_frames_out)            
            insertRow = pd.DataFrame([[video.replace('.avi', ''), first_frame, det_frame_ratio_in, det_frames_out]])
            
            table = pd.concat([table, insertRow], ignore_index = True)
            print(table)
        table.to_csv(self.save_file, header=['video_name', 'first_frame', 'dets_in_anno', 'dets_out_anno'], index = None)
    

class parser_summary(object):
    def __init__(self) -> None:
        super().__init__()
        self.keys = ['time', 'all_fps', 'start', 'det_Pro', 'erosive', 'ulcer', 'other']
    
    def open_summary(self, summary_file):
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        self.summary = dict()
        for line in lines:
            line = line.strip()
            key_value = line.split(' ')
            if len(key_value) == 1:
                if key_value[0] in self.keys:
                    continue
                self.summary[key_value[0].replace('.avi', '')] = dict()
                current_key = key_value[0].replace('.avi', '')
            else:
                self.summary[current_key][key_value[0]] = key_value[1:]
        print(self.summary)
        

class CSV_helper_gastric(object):
    def __init__(self):
        pass

    def open_csv(self, path):
        assert os.path.isfile(path), 'file not exist: {}'.format(path)
        # self.open_xlsx(path)
        self.dataframe = pd.read_csv(path)

    def get_annos(self):
        # print(self.dataframe)
        self.video_names = self.dataframe['video_name']
        self.tp_annos = []
        # import pdb
        # pdb.set_trace()
        for index, name in enumerate(self.video_names):
            # if index <= 25:
            #     continue

            if name[:-4] != '.avi':
                name = name + '.avi'

            tp = {
                'video_name':name,
                'tp_range':[],
                'time_break':[],
                'time_break_name':[]
            }
            
            if not pd.isna(name):

                row = self.dataframe.iloc[index][2:]

                for i in tqdm(range(len(row))):

                    if not (pd.isna(row[i])):
                        row_cell =  row[i].strip().split('-')
                        # print(row_cell, (isinstance(row_cell[0], datetime.time) and isinstance(row_cell[1], datetime.time)))
                        # import pdb
                        # pdb.set_trace()
                        # if (isinstance(row_cell[0], datetime.time) and isinstance(row_cell[1], datetime.time)):
                        try:
                            start = datetime.datetime.strptime(row_cell[0], '%M:%S').time()
                            end = datetime.datetime.strptime(row_cell[1], '%M:%S').time()
                        except:
                            try:
                                start = datetime.datetime.strptime(row_cell[0], '%H:%M:%S').time()
                                end = datetime.datetime.strptime(row_cell[1], '%H:%M:%S').time()
                            except:
                                continue

                        start_second = (start.hour * 60 + start.minute) * 60 + start.second
                        end_second = (end.hour * 60 + end.minute) * 60 + end.second
                        tp['tp_range'].append([start_second, end_second])
                        tp['time_break'].extend([start_second, end_second])
                        tp['time_break_name'].extend([row_cell[0], row_cell[1]])

                self.tp_annos.append(tp)
        # print(self.tp_annos)
        return self.tp_annos



def merge_detection_json(file_paths, new_file='merged.json'):
    jsons = []
    for file in file_paths:
        with open(file, 'r') as f:
            jsons.append(json.load(f))

    merged = dict()
    for j in jsons:
        for detector_name, detector_result in j.items():
            if detector_name in merged:
                old_result = merged[detector_name]
            else:
                old_result = dict()
            for video_name,result in detector_result.items():
                # TODO: remove this in future
                #check if in a mistake format
                if len(video_name.split('/'))>2:
                    video_name = os.path.join(*video_name.split('/')[-2:])
                old_result[video_name] =  result
            merged[detector_name] = old_result

    
    with open(new_file, 'w') as f:
        json.dump(merged, f, indent=4)

    return merged

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="video_evaluation")
    parser.add_argument('--csv_file', type=str, default='datas/anno1103.csv', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_path', type=str, default='datas/', help='source')  # file/folder, 0 for webcam
    args = parser.parse_args()
    evaluator = Evaluator(args)
    csv = CSV_helper_gastric()
    csv.open_csv(args.csv_file)
    gt = csv.get_annos()

    print('total video {}'.format(len(gt)))
    # print(gt)

    evaluator.plot_tables(gt)


