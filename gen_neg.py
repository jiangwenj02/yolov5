import os
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
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import torch
from pathlib import Path

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
        self.saving_root = "/data3/zzhang/tmp/erosive_ulcer_videos/"
        self.video_root = "/data1/qilei_chen/DATA/erosive_ulcer_videos/"

    def _init_detector(self):
        # Initialize
        set_logging()
        device = select_device(self.opt.device)
        self.device = device
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(self.opt.weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(self.opt.imgsz, s=stride)  # check img_size
        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        self.half = half
        if half:
            model.half()  # to FP16
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        self.model = model

    def test_video(self,  csv_gt_annos,start_video_index,end_index):
        self._init_detector()
        print('I am work at video form index {} to index {} '.format(start_video_index, end_index-1))
        csv_videos_susection =  csv_gt_annos[start_video_index:end_index]
        dechun_result = dict()
        for index,  temp in enumerate(csv_videos_susection):
            video, anno = temp['video_name'], temp['tp_range']
            print('current work at: {}'.format(video))
            counter = 0
            video_path = os.path.join(self.video_root, video)
            if not os.path.isfile(video_path):
                print('{} not exist'.format(video_path))
                continue
            dataset = LoadImages(video_path, img_size=self.opt.imgsz, stride=self.opt.stride)
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms,
                                        max_det=self.opt.max_det)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    save_path = str(self.saving_root / p.name)  # img.jpg
                    txt_path = str(self.saving_root / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if self.opt.save_crop else im0  # for opt.save_crop
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # if save_txt:  # Write to file
                            #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            #     with open(txt_path + '.txt', 'a') as f:
                            #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            # if save_img or self.opt.save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.opt.hide_labels else (self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=self.opt.line_thickness)
                            if self.opt.save_crop:
                                save_one_box(xyxy, imc, file=self.saving_root / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')

                    # Save results (image with detections)
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    # def test_video2(self, csv_gt_annos,start_video_index,end_index):
    #     self._init_detector()
    #     print('I am work at video form index {} to index {} '.format(start_video_index, end_index-1))
    #     csv_videos_susection =  csv_gt_annos[start_video_index:end_index]
    #     dechun_result = dict()
    #     for index,  temp in enumerate(csv_videos_susection):
    #         video, anno = temp['video_name'], temp['tp_range']
    #         print('current work at: {}'.format(video))
    #         counter = 0
    #         video_path = os.path.join(self.video_root, video)
    #         if not os.path.isfile(video_path):
    #             print('{} not exist'.format(video_path))
    #             continue
    #         frame_reader, save_result_dir = self._get_frame_reader(video_path,
    #                                                                "evaluation_video")
    #         img = frame_reader.cap_video()
    #         total_frame = frame_reader.get_total_frame_count()
    #         fps = frame_reader.get_fps()
    #         video_writer = cv2.VideoWriter(os.path.join(self.saving_root,video),
    #                                cv2.VideoWriter_fourcc(*'mp4v'),
    #                                frame_reader.get_fps(),
    #                                (img.shape[1], img.shape[0]), True)

    #         img_fp_folder = os.path.join(self.saving_root, video.split('.av')[0])
    #         os.makedirs(img_fp_folder, exist_ok=True)
    #         dechun_video_result = {
    #             'total_frame': total_frame,
    #             'fps': fps,
    #             'size': img.shape[:2],
    #             'detection_result': []
    #         }

    #         pbar = tqdm(total=total_frame)
    #         pbar.update(1)

    #         while img is not None:
    #             out_img = img[:,:,(2,1,0)].copy()

    #             all_detector_result = self.polyp_detector.get_results(img)
    #             frame_time = counter / float(fps)

    #             #dechun
    #             if len(all_detector_result[0])>0:
    #                 result = {}
    #                 result['bbox'] = [a.tolist() for a in all_detector_result[0]]
    #                 result['time'] = frame_time
    #                 result['frame_counter'] = counter
    #                 dechun_video_result['detection_result'].append(result)

    #                 for box in all_detector_result[0]:
    #                     if box[4] >= 0.5:
    #                         pt1 = (int(box[0]), int(box[1]))
    #                         pt2 = (int(box[2]), int(box[3]))
    #                         text_pt = (int(box[0]), int(box[1] - 10))
    #                         label = int(box[5])-1
    #                         cv2.rectangle(out_img, pt1, pt2,  COLOR[label], 2)
    #                         cv2.putText(out_img, f'{LABEL[label]}_{box[4]:.4f}', text_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR[label], 2)
    #                 if not time_in_list_range(anno,frame_time)[0]:
    #                     cv2.imwrite(os.path.join(img_fp_folder,f"{video}_{counter}_dw.jpg"),out_img)
    #                     cv2.imwrite(os.path.join(img_fp_folder, f"{video}_{counter}.jpg"), img[:,:,(2,1,0)])

    #             video_writer.write(out_img)
    #             pbar.update(1)
    #             img = frame_reader.cap_video()
    #             counter += 1

    #         dechun_result[video] = dechun_video_result
    #         pbar.close()
    #         print(' Done {}/{}'.format(index + 1, len(csv_videos_susection)))
    #         out= {
    #             'dechun_result': dechun_result,

    #         }
    #         saving_path = os.path.join(self.saving_root, 'all_result_{}_{}.json'.format(start_video_index,end_index))
    #         with open(saving_path, 'w') as f:
    #             json.dump(out, f, indent=4)
    #         print('result file save at: {}'.format(saving_path))




    # def _get_frame_reader(self, video_dir, mode):
    #     abs_path = video_dir  # directory of each video
    #     assert os.path.isfile(abs_path), print('file does not exist: {}'.format(abs_path))

    #     file_name_with_extention = os.path.basename(abs_path)
    #     file_name = file_name_with_extention.split('.')[0]
    #     self.file_name = file_name
    #     save_result_dir = os.path.join(self.saving_root, mode, file_name)
    #     print('save anno at dir: {}'.format(save_result_dir))
    #     os.makedirs(save_result_dir, exist_ok=True)

    #     if len(file_name) >= 10:
    #         roi = rois['small']
    #     else:
    #         roi = rois['big']

    #     if file_name in rois:
    #         roi = rois[file_name]

    #     frame_reader = FrameLoader(roi=[
    #                           218,
    #                           47,
    #                           693,
    #                           534
    #                          ])
    #     frame_reader.open_video(abs_path)

    #     print('current work with: {}'.format(file_name_with_extention))
    #     return frame_reader, save_result_dir

    def _save_results(self, img, polyp_result, save_result_dir, counter):
        self.polyp_detector.draw_rect(img, polyp_result)
        imageio.imwrite(os.path.join(save_result_dir, str(counter) + '.jpg'), img)

    def time_in_list_range(self, range_list, x):
        for i, range in enumerate(range_list):
            if self.time_in_range(range[0], range[1], x):
                return True, i

        return False, 0

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
                'tp_range':[]
            }
            if not pd.isna(name):

                row = self.dataframe.iloc[index][3:]

                i = 0
                while i < len(row):

                    if not (pd.isna(row[i])):
                        row_cell =  row[i].split('-')
                        if not  (isinstance(row_cell[0], datetime.time) and isinstance(row_cell[1], datetime.time)):
                            try:

                                start = datetime.datetime.strptime(row_cell[0], '%M:%S').time()
                                end = datetime.datetime.strptime(row_cell[ 1], '%M:%S').time()

                            except:

                                start = datetime.datetime.strptime(row_cell[0], '%H:%M:%S').time()
                                end = datetime.datetime.strptime(row_cell[ 1], '%H:%M:%S').time()
                        else:
                            start = row_cell[0]
                            end = row_cell[1]


                        start_second = (start.hour * 60 + start.minute) * 60 + start.second
                        end_second = (end.hour * 60 + end.minute) * 60 + end.second
                        tp['tp_range'].append([start_second, end_second])

                    i += 2
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
    parser.add_argument('--start', default=0, type=int,  help="video index to start")
    parser.add_argument('--end', default=0, type=int, help="video index to end")
    args = parser.parse_args()
    # evaluator = Evaluator()
    csv = CSV_helper_gastric()
    csv.open_csv('neg.csv')

    gt = csv.get_annos()
    print('total video {}'.format(len(gt)))
    if args.end == 0:
        args.end = len(gt)
    start_video_index = args.start
    end_index = args.end
    print(gt)

    # evaluator.test_video2(gt,start_video_index,end_index)


