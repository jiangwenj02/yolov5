python test2.py --weights runs/train/exp4/weights/best.pt --save-json --data data/coco.yaml --ann /data3/zzhang/annotation/erosiveulcer_fine/test.json
python gen_neg.py --csv_file /data3/zzhang/tmp/anno0724.csv --video_path /data2/qilei_chen/DATA/erosive_ulcer_videos/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos0712_th03_448/ --weights runs/train/exp15/weights/best.pt --conf-thres 0.5
python gen_neg.py --csv_file /data3/zzhang/annotation/erosiveulcer_fine/neg0615.csv --video_path /data3/zzhang/tmp/videos/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos0712_th03/ --weights runs/train/exp14/weights/best.pt --conf-thres 0.5 
python gen_neg_cls.py --csv_file /data3/zzhang/tmp/anno0724.csv --video_path /data2/qilei_chen/DATA/erosive_ulcer_videos/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos0831_cls/  --weights /data3/zzhang/mmclassification/work_dirs/resnet50_cancer_adddata/latest.pth


####adding new data
1、ls images/ > new.txt
2、mmdetection python tools/dataset_converters/image2json.py
3、 sed -e 's#^#/data3/zzhang/tmp/Others/Others_test/#' /data3/zzhang/tmp/Others/test.txt > /data3/zzhang/tmp/Others/test1006.txt
4、 modify cocoempty.yaml
5、 python test2.py --weights runs/train/exp21/weights/best.pt --save-json --data data/cocoempty.yaml --ann /data3/zzhang/tmp/Others/fp.json --conf-thres 0.5
6、python tools/merge_diff_json.py
7、python tools/dataset_converters/coco_convert_darknet.py