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
    def __init__(self, lib_dir, weights, save_dir=None):
        # Load model
        self.model = torch.hub.load(lib_dir, 'custom', path=weights, source='local', force_reload=True)
        self.save_dir = save_dir

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
        pred = self.model(image)
        if self.save_dir is not None:
            pred.save()
        results = dict()
        for i in range(pred.xyxy[0].shape[0]):
            results[i] = dict()
            results[i]['score'] = pred.xyxy[0][i, 4]
            results[i]['pt'] = pred.xyxy[0][i, 0:4]
            results[i]['type'] = pred.xyxy[0][i, 5]
        return results

    def __call__(self, image: np.ndarray):
        return self.predict(image)


if __name__ == '__main__':
    detector = YoloBase('./', './runs/train/exp4/weights/best.pt', save_dir='runs/hub/exp')
    detector.predict('/home1/users/jiangwenj02/mmdetection/data/erosiveulcer/images/00088057-49db-4200-ad48-4a011b0ff906.jpg')