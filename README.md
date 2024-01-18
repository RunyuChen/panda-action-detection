# panda-action-detection
chenrunyu59@gmail.com

采用FasterRcnn + slowfast进行动作检测

# det_FasterRcnn

Faster Rcnn检测模型

## 1 detect_img.py

调用模型进行检测

可选的参数在文件里有写

## 2 detect_img.sh

对一个文件夹中的图片进行检测的示例

运行：

```
bash detect_img.sh
```

需要将文件中的路径换成实际路径

## 3 demo_action_detect.py

行为识别的检测模型（detect.py）

执行：

```
python demo/detect.py --config configs/detection/ava/myslowfast.py --checkpoint /home/ubuntu/panda_videofile/best_mAP@0.5IOU_epoch_19.pth --det-config demo/faster_rcnn_r50_fpn_2x_coco.py  --det-config demo/faster_rcnn_r50_fpn_2x_coco.py（替换为新的检测配置文件） --det-checkpoint /home/ubuntu/LSQ/detection2/mmaction2_YF/Checkpionts/mmdetection/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth（替换成新的检测模型）   --video /home/ubuntu/panda_videofile/long.mp4  --out-filename /home/ubuntu/panda_videofile/long_det.mp4   --det-score-thr 0.2 --action-score-thr 0.2 --output-stepsize 8 --predict-stepsize 30  --output-fps 6 --label-map tools/data/ava/label_map2.txt
```

