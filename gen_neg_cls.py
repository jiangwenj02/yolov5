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
from mmcls.apis import init_model, inference_model
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
        self.det_summary = osp.join(self.saving_root, 'summary.txt')
        self.save_train_images = opt.save_train_images
        os.popen('rm -r ' + osp.join(self.saving_root, '*'))
        os.makedirs(self.saving_root, exist_ok=True)

    def _init_detector(self):

        device = select_device(self.opt.device)
        self.device = device

        # Load model
        model = init_model(self.opt.config, checkpoint=self.opt.weights[0], device='cpu')
        self.names = model.CLASSES
        self.model = model.to(self.device)

    def test_video(self,  csv_gt_annos, start_video_index, end_index):
        self._init_detector()
        print('I am work at video form index {} to index {} '.format(start_video_index, end_index-1))
        csv_videos_susection = csv_gt_annos[start_video_index:end_index]
        summary_f = open(self.det_summary, 'w')

        for index,  temp in enumerate(csv_videos_susection):
            video, anno, break_time, break_time_name = temp['video_name'], temp['tp_range'], temp['time_break'], temp['time_break_name']
            object_count = [[0] * len(self.names)] * (len(temp['time_break']) + 1)
            object_count = np.array(object_count)
            all_fps_count = [0] * (len(temp['time_break']) + 1)
            det_fps_count = [0] * (len(temp['time_break']) + 1)
            start_fps_count = [0] * (len(temp['time_break']) + 1)
            print('current work at: {}'.format(video))

            video_path = os.path.join(self.video_root, video)
            vid_path, vid_writer = None, None
            if not os.path.isfile(video_path):
                print('{} not exist'.format(video_path))
                continue
            cap = cv2.VideoCapture(video_path)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = tqdm(range(length))
            count = 0 
            start_time = time.time()
            p = Path(video_path)  # to Path
            save_path = osp.join(self.saving_root, p.stem)  # img.jpg
            print(save_path)
            vid_save_path = osp.join(save_path, p.stem)
            fp_save_dirs = []
            for name in self.names:
                fp_save_dirs.append(osp.join(save_path, name))
                os.makedirs(osp.join(save_path, name), exist_ok=True)
            os.makedirs(save_path, exist_ok=True)
            for frame in pbar:
                torch.cuda.empty_cache()
                ret_val, img_ori = cap.read()
                if not ret_val:
                    break
                crop_imgs = crop_img(img_ori)

                # Inference
                t1 = time_synchronized()
                result = inference_model(self.model, crop_imgs)
                img = self.model.show_result(img_ori, result, show=False)
                t2 = time_synchronized()
                
                frame_time = frame / float(fps)
                # import pdb
                # pdb.set_trace()                

                time_idx = self.time_in_break_time(break_time, frame_time)
                all_fps_count[time_idx]  += 1
                print(result['pred_label'], self.save_train_images)
                if result['pred_label'] == 0 and self.save_train_images is not None:
                    print(os.path.join(self.save_train_images, f"{video}_{frame}.jpg"))
                    cv2.imwrite(os.path.join(self.save_train_images, f"{video}_{frame}.jpg"), crop_imgs) 
                cv2.imwrite(os.path.join(fp_save_dirs[result['pred_label']], f"{video}_{frame}.jpg"), img)                

                if vid_path != vid_save_path:  # new video
                    w, h = img.shape[1], img.shape[0]
                    vid_save_path += '.mp4'  
                    vid_path = vid_save_path                                          
                    vid_writer = cv2.VideoWriter(vid_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                img = mmcv.imresize(img, size=(frame_width, frame_height))
                vid_writer.write(img)
                count = count + 1
 
            end_time = time.time()
            spend_time = (end_time - start_time)
            det_speed = frame / spend_time
            det_fps_count = [round((100 * det_fps_count[i] / (all_fps_count[i]+0.1)), 2) for i in range(len(all_fps_count))]
            summary_f.write(video + '\n')
            summary_f.write('fps: %.2f' % (det_speed) + '\n')
            summary_f.write('time ' + ' '.join(break_time_name) + '\n')
            summary_f.write('all_fps ' + ' '.join([str(item) for item in all_fps_count]) + '\n')
            summary_f.write('start ' + ' '.join([str(item) for item in start_fps_count]) + '\n')
            summary_f.write('det_Pro ' + ' '.join([str(item) for item in det_fps_count]) + '\n')
            for i in range(len(self.names)):
                summary_f.write(self.names[i] + ' ' + ' '.join([str(item) for item in object_count[:, i].tolist()]) + '\n')
        summary_f.close()

    def time_in_list_range(self, range_list, x):
        for i, range in enumerate(range_list):
            if self.time_in_range(range[0], range[1], x):
                return True, i

        return False, 0

    def time_in_break_time(self, break_time, x):
        for i, range in enumerate(break_time):
            if x <= range:
                return i
        return len(break_time)

    def time_in_range(self, start, end, x):
        """Return true if x is in the range [start, end]"""
        if start <= end:
            return start <= x <= end
        else:
            return start <= x or x <= end


class CSV_helper(object):
    def __init__(self):
        pass

    def open_csv(self, path):
        assert os.path.isfile(path), 'file not exist: {}'.format(path)
        self.dataframe = pd.read_csv(path)

    def get_annos(self):

        self.video_names = self.dataframe['video name']
        self.tp_annos = []

        for index, name in enumerate(self.video_names):
            tp = {
                'video_name':name,
                'tp_range':[]
            }
            if not pd.isna(name):

                row = self.dataframe.iloc[index][5:]

                i = 0
                while i < len(row):

                    if not (pd.isna(row[i]) or pd.isna(row[i + 1])):
                        try:

                            start = datetime.datetime.strptime(row[i], '%M:%S').time()
                            end = datetime.datetime.strptime(row[i + 1], '%M:%S').time()

                        except:

                            start = datetime.datetime.strptime(row[i], '%H:%M:%S').time()
                            end = datetime.datetime.strptime(row[i + 1], '%H:%M:%S').time()

                        start_second = (start.hour * 60 + start.minute) * 60 + start.second
                        end_second = (end.hour * 60 + end.minute) * 60 + end.second
                        tp['tp_range'].append([start_second, end_second])

                    i += 5
                self.tp_annos.append(tp)
        return self.tp_annos

class CSV_helper_gastric(object):
    def __init__(self):
        pass

    def open_csv(self, path):
        assert os.path.isfile(path), 'file not exist: {}'.format(path)
        # self.open_xlsx(path)
        self.dataframe = pd.read_csv(path)
    def open_xlsx(self, path):
        assert os.path.isfile(path), 'file not exist: {}'.format(path)
        self.dataframe = pd.read_excel(path)


    def get_annos(self):
        print(self.dataframe)
        self.video_names = self.dataframe['video_name']
        self.tp_annos = []

        for index, name in enumerate(self.video_names):
            tp = {
                'video_name':name,
                'tp_range':[],
                'time_break':[],
                'time_break_name':[]
            }
            if not pd.isna(name):

                row = self.dataframe.iloc[index][2:]

                i = 0
                while i < len(row):

                    if not (pd.isna(row[i])):
                        row_cell =  row[i].strip().split('-')
                        if not  (isinstance(row_cell[0], datetime.time) and isinstance(row_cell[1], datetime.time)):
                            try:
                                start = datetime.datetime.strptime(row_cell[0], '%M:%S').time()
                                end = datetime.datetime.strptime(row_cell[1], '%M:%S').time()

                            except:

                                start = datetime.datetime.strptime(row_cell[0], '%H:%M:%S').time()
                                end = datetime.datetime.strptime(row_cell[ 1], '%H:%M:%S').time()
                        else:
                            start = row_cell[0]
                            end = row_cell[1]


                        start_second = (start.hour * 60 + start.minute) * 60 + start.second
                        end_second = (end.hour * 60 + end.minute) * 60 + end.second
                        tp['tp_range'].append([start_second, end_second])
                        tp['time_break'].extend([start_second, end_second])
                        tp['time_break_name'].extend([row_cell[0], row_cell[1]])

                    i += 1
                self.tp_annos.append(tp)
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
    parser.add_argument('--csv_file', type=str, default='neg0615.csv', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--video_path', type=str, default='/data2/qilei_chen/DATA/erosive_ulcer_videos', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_path', type=str, default='/data3/zzhang/tmp/erosive_ulcer_videos0615/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_train_images', default=None, help='source')  # /data3/zzhang/tmp/classification/train/non_cancer/
    # parser.add_argument('--det_summary', type=str, default='/data3/zzhang/tmp/erosive_ulcer_videos0615/summary.txt', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--start', default=0, type=int,  help="video index to start")
    parser.add_argument('--end', default=0, type=int, help="video index to end")
    parser.add_argument('--config', nargs='+', type=str, default='/data3/zzhang/mmclassification/configs/diseased/resnet50_cancer.py', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='/data3/zzhang/mmclassification/work_dirs/resnet50_cancer/latest.pth', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    args = parser.parse_args()
    evaluator = Evaluator(args)
    csv = CSV_helper_gastric()
    csv.open_csv(args.csv_file)

    gt = csv.get_annos()
    print('total video {}'.format(len(gt)))
    if args.end == 0:
        args.end = len(gt)
    start_video_index = args.start
    end_index = args.end
    print(gt)

    evaluator.test_video(gt,start_video_index,end_index)


