python test2.py --weights runs/train/exp4/weights/best.pt --save-json --data data/coco.yaml --ann /data3/zzhang/annotation/erosiveulcer_fine/test.json
python gen_neg.py --csv_file /data3/zzhang/tmp/anno0724.csv --video_path /data2/qilei_chen/DATA/erosive_ulcer_videos/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos0712_th03_448/ --weights runs/train/exp15/weights/best.pt --conf-thres 0.5
python gen_neg.py --csv_file /data3/zzhang/annotation/erosiveulcer_fine/neg0615.csv --video_path /data3/zzhang/tmp/videos/ --save_path /data3/zzhang/tmp/erosive_ulcer_videos0712_th03/ --weights runs/train/exp14/weights/best.pt --conf-thres 0.5 