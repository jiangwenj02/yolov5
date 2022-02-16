python test2.py --weights runs/train/exp4/weights/best.pt --save-json --data data/coco.yaml --ann /data3/zzhang/annotation/erosiveulcer_fine/test.json
python gen_neg.py --csv_file /data3/zzhang/tmp/anno0724.csv --video_path /data2/qilei_chen/DATA/erosive_ulcer_videos/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos0712_th03_448/ --weights runs/train/exp15/weights/best.pt --conf-thres 0.5
python gen_neg.py --csv_file /data3/zzhang/annotation/erosiveulcer_fine/neg0615.csv --video_path /data3/zzhang/tmp/videos/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos0712_th03/ --weights runs/train/exp14/weights/best.pt --conf-thres 0.5 
python gen_neg_cls.py --csv_file /data3/zzhang/tmp/anno0724.csv --video_path /data2/qilei_chen/DATA/erosive_ulcer_videos/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos0831_cls/  --weights /data3/zzhang/mmclassification/work_dirs/resnet50_cancer_adddata/latest.pth
python gen_neg.py --csv_file /data3/zzhang/tmp/anno1110-5.csv --video_path /data2/qilei_chen/jianjiwanzhengshipin2/preprocessed/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos1110/preprocessed/ --weights runs/train/exp30/weights/best.pt --conf-thres 0.5 && python gen_neg.py --csv_file /data3/zzhang/tmp/anno1110-5.csv --video_path /data2/qilei_chen/jianjiwanzhengshipin2/preprocessed2/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos1110/preprocessed2/ --weights runs/train/exp30/weights/best.pt --conf-thres 0.5 && python gen_neg.py --csv_file /data3/zzhang/tmp/anno1110-5.csv --video_path /data2/qilei_chen/jianjiwanzhengshipin2/preprocessed/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos1110/preprocessed/ --weights runs/train/exp30/weights/best.pt --conf-thres 0.5
python gen_neg.py --csv_file /data3/zzhang/tmp/anno1110-5.csv --video_path /data2/qilei_chen/jianjiwanzhengshipin2/preprocessed/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos1221/preprocessed/ --weights runs/train/exp34/weights/last.pt --conf-thres 0.5
python gen_neg.py --csv_file /data3/zzhang/tmp/anno1228.csv --video_path /data3/zzhang/tmp/test-video-1226/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos1228/ --weights runs/train/exp35/weights/last.pt --conf-thres 0.5
python gen_neg.py --csv_file /data3/zzhang/tmp/anno_cls_0122.csv --video_path /data3/zzhang/tmp/yanshi --save_path /data3/zzhang/tmp/erosive_ulcer_videos0122/yanshi/ --weights runs/train/exp36/weights/last.pt --conf-thres 0.5
python gen_neg.py --csv_file /data3/zzhang/tmp/anno1228.csv --video_path /data3/zzhang/tmp/test-video-1226/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos0122/erosive_ulcer_videos/ --weights runs/train/exp36/weights/last.pt --conf-thres 0.5
##20220216
python gen_neg.py --csv_file /data3/zzhang/tmp/anno_cls_0122.csv --video_path /data3/zzhang/tmp/yanshi --save_path /data3/zzhang/tmp/erosive_ulcer_videos0216/yanshi/ --weights runs/train/exp39/weights/last.pt --conf-thres 0.5
python gen_neg.py --csv_file /data3/zzhang/tmp/anno1228.csv --video_path /data3/zzhang/tmp/test-video-1226/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos0216/erosive_ulcer_videos/ --weights runs/train/exp39/weights/last.pt --conf-thres 0.5
python gen_neg_images.py --image_path /data3/zzhang/tmp/gastric_3cls_0921/ --save_path /data3/zzhang/tmp/gastric_3cls_0921_det_0216/  --weights runs/train/exp39/weights/last.pt --conf-thres 0.5

####adding new data
1、ls images/ > new.txt
2、mmdetection python tools/dataset_converters/image2json.py
3、 sed -e 's#^#/data3/zzhang/tmp/Others/Others_test/#' /data3/zzhang/tmp/Others/test.txt > /data3/zzhang/tmp/Others/test1006.txt
4、 modify cocoempty.yaml
5、 python test2.py --weights runs/train/exp21/weights/best.pt --save-json --data data/cocoempty.yaml  --conf-thres 0.5 --ann /data3/zzhang/tmp/Others/fp.json
6、 python tools/det2json.py #mmdetection
7、python tools/merge_diff_json.py
8、python tools/dataset_converters/coco_convert_darknet.py
9、modify cocofilter.yaml
10、python train.py --data data/cocofilter.yaml --cfg yolov5s.yaml --weights '' 

####gen video empty anno
python gen_anno.py --dirs /data2/zinan_xiong/gastritis_videos_for_test/ --hz mp4 --out /data3/zzhang/tmp/anno_cls_0104.csv
python gen_neg_cls.py --csv_file /data3/zzhang/tmp/anno_cls_0125.csv  --video_path /data2/zinan_xiong/gastritis_videos_for_test_2/ \
--save_path ../mmclassification/work_dirs/shuffle_stomach_mix_hr/gastritis_videos_for_test2_0201/ \
--weights /data3/zzhang/mmclassification/work_dirs/shuffle_stomach_mix_hr/latest.pth \
--config /data3/zzhang/mmclassification/configs/diseased/shuffle_stomach_mix_hr.py

python gen_neg_cls.py --csv_file /data3/zzhang/tmp/anno_cls_0125.csv  --video_path /data2/zinan_xiong/gastritis_videos_for_test_2/ \
--save_path ../mmclassification/work_dirs/seresnet50_b32x8_stomach_mix_hr/gastritis_videos_for_test2_0201/ \
--weights /data3/zzhang/mmclassification/work_dirs/seresnet50_b32x8_stomach_mix_hr/latest.pth \
--config /data3/zzhang/mmclassification/configs/diseased/seresnet50_b32x8_stomach_mix_hr.py