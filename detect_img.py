'''
This file is used to detect the pictures in the folder
data: 2024/1/17
'''
import os
import asyncio
import cv2
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file or directory')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out-dir', default=None, help='Path to output directory')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='Whether to set async options for async inference.')
    parser.add_argument(
        '--output-img',
        action='store_true',
        help='Whether to output images with bounding boxes')
    parser.add_argument(
        '--output-yolo',
        action='store_true',
        help='Whether to output YOLO format text files')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    if os.path.isdir(args.img):
        # If input is a directory, process all images in the directory
        image_list = [
            os.path.join(args.img, img) for img in os.listdir(args.img)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    else:
        image_list = [args.img]

    for img_path in image_list:
        # test each image
        result = inference_detector(model, img_path)
        
        # save the results
        save_path = os.path.join(
            args.out_dir, f'result_{os.path.basename(img_path)}')
        if args.output_img:
            show_result_pyplot(
                model,
                img_path,
                result,
                palette=args.palette,
                score_thr=args.score_thr,
                out_file=save_path)

        # Save YOLO format text file if specified
        if args.output_yolo:
            yolo_txt_path = os.path.join(
                args.out_dir, f'result_{os.path.splitext(os.path.basename(img_path))[0]}.txt')
            original_img = cv2.imread(img_path)
            img_height, img_width = original_img.shape[:2]
            save_yolo_format(result, yolo_txt_path, img_height, img_width)


def save_yolo_format(result, save_path, img_height, img_width):
    # Implement the logic to convert detection results to YOLO format and save to file
    with open(save_path, 'w') as f:
        print(result)
        print(result[0])
        for bbox in result[0]:
            # Assuming bbox is in the format [x_min, y_min, x_max, y_max]
            dh = 1./ (img_height)
            dw = 1./ (img_width)
            x_center = (bbox[0] + bbox[2]) / 2.0 - 1
            y_center = (bbox[1] + bbox[3]) / 2.0 -1
            width = (bbox[2] - bbox[0]) 
            height = (bbox[3] - bbox[1]) 
            x_center = x_center * dw
            y_center = y_center * dh
            width = width * dw
            height = height * dh
            score = bbox[4]
            f.write(f'0 {x_center} {y_center} {width} {height} {score}\n')


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    if os.path.isdir(args.img):
        # If input is a directory, process all images in the directory
        image_list = [
            os.path.join(args.img, img) for img in os.listdir(args.img)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    else:
        image_list = [args.img]

    tasks = [async_inference_detector(model, img) for img in image_list]
    results = await asyncio.gather(*tasks)

    for img_path, result in zip(image_list, results):
        # save the results
        save_path = os.path.join(
            args.out_dir, f'result_{os.path.basename(img_path)}')
        show_result_pyplot(
            model,
            img_path,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=save_path)

        # Save YOLO format text file if specified
        if args.output_yolo:
            yolo_txt_path = os.path.join(
                args.out_dir, f'result_{os.path.splitext(os.path.basename(img_path))[0]}.txt')
            save_yolo_format(result, yolo_txt_path)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
