# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta

import torch
import numpy as np
from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
import cv2
from utils.datasets import letterbox

class YoloBase(metaclass=ABCMeta):
    def __init__(self, weights, imgsz=600, conf_thres=0.30, half=False, device='', iou_thres=0.45, max_det=1000, augment=False, agnostic_nms=False, classes=None):
        # Load model
        device = select_device(device)
        model = torch.hub.load('./', 'custom', path='./runs/train/exp4/weights/best.pt', source='local', force_reload=True)
        pred = model('/home1/users/jiangwenj02/mmdetection/data/erosiveulcer/images/00088057-49db-4200-ad48-4a011b0ff906.jpg')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size        
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        self.model = model
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.half = half
        self.device = device
        self.names = names
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.stride = stride
        self.img_size = imgsz


    @abstractmethod
    def predict(self, image: np.ndarray):
        """
        图像AI分析

        Args:
            image (np.ndarray): 待分析的图像

        Returns:
            dict: 返回分析结果数据

            示例:
            {
                0: {
                    'score': tensor(0.5932),
                    'pt': tensor([353.1492, 144.6515, 424.8212, 248.7242]),
                    'type': 'AP'
                },
                1: { ... }
            }
            0：表示当前息肉编号，后面即为该息肉对应的数据；
            score：为该息肉分数，分数越高表示息肉可能性越大；
            pt：为息肉坐标；
            type：为息肉类型：AP(腺瘤性息肉)、IP（炎性息肉）、HP（错构瘤性息肉）及OP（其他非上述类型息肉）

        """
        if type(image) is str:
            image = cv2.imread(image)
        h0, w0 = image.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(image, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        h, w = img.shape[:2]  # img, hw_original, hw_resized

        img, ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(image).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)        

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        t2 = time_synchronized()

        predn = pred.clone()
        scale_coords(img.shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

    def __call__(self, image: np.ndarray):
        return self.predict(image)
