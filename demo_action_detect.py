# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import os
import os.path as osp
import shutil
import time
import gc

import imageio.v2 as iio
import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from mmaction.models import build_detector
from mmaction.utils import import_module_error_func

from collections import Counter

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    '''此处最好不要掩藏报错, 应该直接抛出异常'''
    raise ImportError('Please install mmdet')
    # @import_module_error_func('mmdet')
    # def inference_detector(*args, **kwargs):
    #     pass

    # @import_module_error_func('mmdet')
    # def init_detector(*args, **kwargs):
    #     pass

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

# 全局变量, 记录各个函数的执行耗时, 用于最后的输出
g_all_func_cost = []


# 统计函数的执行耗时
class TimeCostDecorator():
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        st = time.time()
        ret = self.func(*args, **kwargs)
        cost = time.time() - st
        print("\n**** timecost of %s: %.3f seconds\n" % (self.func.__name__, cost))
        g_all_func_cost.append((self.func.__name__, cost))
        return ret


# 打印所有记录起来的执行耗时
def print_all_cost():
    """Print all time cost of function recorded in global var:
    g_all_func_cost
    """

    print("all cost time info (seconds)")
    for c in g_all_func_cost:
        print("%25s : %.3f" % (c[0], c[1]))


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]

# 得到检测框中得分最高的类别
def get_best_action(results):
    new_results = []

    for result in results:
        if result is None:
            new_results.append(None)
            continue
        max_index = 0
        max_score = 0
        max_label = 0
        for i in range(len(result)):
            # 获取当前元素中的行为标签和对应的得分
            boxes, labels, scores = result[i][0], result[i][1], result[i][2]
            # 找到得分最高的行为标签
            max_score_index = np.argmax(scores)
            max_score_label = labels[max_score_index]
            if scores[max_score_index] > max_score:
                max_score = scores[max_score_index]
                max_index = i
                max_label = labels[max_score_index]
        # 构建新的元素，将行为标签替换为得分最高的标签
        new_result = (boxes, [max_label], [max_score])
        new_results.append(new_result)

    return new_results

# 滑动窗口后处理
def postprocess(new_results, predict_stepsize):
    assert 128 % predict_stepsize == 0  # 128根据所需的滑动窗口大小进行修改
    # print('n=', 64 / predict_stepsize)
    n = int(128 / predict_stepsize)  # 128根据所需的滑动窗口大小进行修改
    result_with_most_common = []
    rest = new_results[0:n]
    for i in range(n, len(new_results)):
        # 构建滑动窗口
        window = new_results[i - n:i]

        # 统计行为标签出现次数
        all_labels = []
        for item in window:
            if item == None:
                all_labels.append(None)
                continue
            all_labels.append(item[1][0])
        # print(all_labels)
        label_counts = Counter(all_labels)
        # print('\nlabel_counts:\n',label_counts)

        # 获取出现次数最多的行为标签
        most_common_label = label_counts.most_common(1)[0][0]
        # print('\nmost_common_label:\n',most_common_label)

        # 将第n个元素的行为标签替换为出现次数最多的行为标签
        if most_common_label == None or window[-1]== None:
            result_with_most_common.append(None)
        else:
            result_with_most_common.append((window[-1][0], [most_common_label], window[-1][2]))

    return rest + result_with_most_common


def visualize(frames, annotations, predict_stepsize, plate=plate_blue):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.

    Returns:
        list[np.ndarray]: Visualized frames.
    """
    annotations = get_best_action(annotations)
    annotations = postprocess(annotations, predict_stepsize)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])
    for i in range(na):
        anno = annotations[i]
        print('anno:', anno)
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]
            box = anno[0]
            label = anno[1]
            score = anno[2]
            box = (box * scale_ratio).astype(np.int64)
            st, ed = tuple(box[:2]), tuple(box[2:])
            cv2.rectangle(frame, st, ed, plate[0], 2)


            text = str(label)
            #text = ': '.join([text, str(score[k])])
            location = (0 + st[0], 18 + st[1])
            textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                       THICKNESS)[0]
            textwidth = textsize[0]
            diag0 = (location[0] + textwidth, location[1] - 14)
            diag1 = (location[0], location[1] + 2)
            cv2.rectangle(frame, diag0, diag1, plate[1], -1)
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)

    return frames_


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--config',
        default=('configs/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py'),
        help='spatio temporal detection config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.5,
        help='the threshold of human action score')
    parser.add_argument('--video', help='video file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/ava/label_map.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--out-filename',
        default='demo/stdet_demo.mp4',
        help='output filename')
    parser.add_argument(
        '--out-txtname',
        default='demo/stdet_demo_result.txt',
        help='output txtname')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=4,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=6,
        type=int,
        help='the fps of demo video output')
    parser.add_argument(
        '--visualize',
        default=0,
        type=int,
        help='show the visualized demo, (1 for yes, 0 for no)')
    parser.add_argument(
        '--show-progress-bar',
        default=1,
        type=int,
        help='show progress bar(1 for yes, 0 for no)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


@TimeCostDecorator
def frame_extraction(video_path, show_progress_bar):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    print('flag', flag)
    cnt = 0
    while flag:
        if show_progress_bar == 1:
            print("\r frame_extraction: " + str(cnt), end="")
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        '''[效率优化] 后续操作都直接使用已经加载到内存的 frames, 以提高运行速度.
        cv2.imwrite 或 cv2.imread 都需要磁盘操作, 会很慢, 所以此处干脆不写文件.
        '''
        # cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames


@TimeCostDecorator
def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference,
           or loaded images.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    if args.show_progress_bar == 1:
        prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        if args.show_progress_bar == 1:
            prog_bar.update()
    print("\n")  # 输出换行, 避免输出混在一起
    return results


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.

    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


@TimeCostDecorator
def write_result_txt(txt_path, results):
    with open(txt_path, "w") as file:
        for annotation in results:
            if annotation is not None:
                for annotation_tuple in annotation:
                    # 将元组中的数据转换为字符串并写入文件
                    annotation_str = " ".join(map(str, annotation_tuple))
                    file.write(annotation_str + "\n")
            else:
                file.write("None\n")
    print("\n检测结果已保存到", txt_path)


@TimeCostDecorator
def write_visualize_video(args, timestamps, results, original_frames):
    print('Performing visualization')

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int_)

    assert args.predict_stepsize % args.output_stepsize == 0
    dense_n = int(args.predict_stepsize / args.output_stepsize)

    '''[效率优化] 使用已经加载到内存的 original_frames, 以提高运行速度.
    '''
    # frames = [
    #     cv2.imread(frame_paths[i - 1])
    #     for i in dense_timestamps(timestamps, dense_n)
    # ]
    frames = [
        original_frames[i - 1]
        for i in dense_timestamps(timestamps, dense_n)
    ]
    vis_frames = visualize(frames, results, args.predict_stepsize)

    '''方式1: 使用 moviepy 来保存视频文件.
    速度慢, 但会调用 ffmpeg 进行视频压缩.
    '''

    def write_video_by_moviepy(vidname):
        print("write_video_by_moviepy begin")
        vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                    fps=args.output_fps)
        vidlogger = None
        if args.show_progress_bar == 1:
            vidlogger = 'bar'
        vid.write_videofile(vidname, logger=vidlogger)
        print("\n视频已使用 moviepy 保存到:", vidname)

    '''方式2: 使用 cv2 来保存视频文件.
    速度快, 虽然不像 moviepy 那样会调用 ffmpeg 对视频进行压缩会导致视频体积大一倍左右, 
    但是用于展示的场景, 我们更关心运行速度, 不太需要考虑体积问题; 另外, 在生成熊猫的个体报告时, 
    是不需要 visualize 的, 只需要拿到 results 进行统计即可.
    '''

    def write_video_by_cv2(vidname):
        print("write_video_by_cv2 begin")
        fps = args.output_fps
        h, w, _ = vis_frames[0].shape
        size = (w, h)
        # 编码为 mp4v 格式
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid = cv2.VideoWriter(vidname, fourcc, fps, size, isColor=True)
        if args.show_progress_bar == 1:
            prog_bar = mmcv.ProgressBar(len(vis_frames))
        for x in vis_frames:
            vid.write(x)
            if args.show_progress_bar == 1:
                prog_bar.update()
        vid.release()
        print("\n视频已使用 cv2 保存到:", vidname)

    '''方式3: 使用 imageio 来保存视频文件.
    速度比 moviepy 快，且能调用 ffmpeg 生成 h264 编码的视频.

    安装 imageio: conda install imageio
    '''

    def write_video_by_imageio(vidname):
        print("write_video_by_imageio begin")
        fps = args.output_fps
        h, w, _ = vis_frames[0].shape
        size = (w, h)
        vid = iio.get_writer(vidname, format='ffmpeg', mode='I', fps=fps, codec='libx264', pixelformat='yuv420p')
        if args.show_progress_bar == 1:
            prog_bar = mmcv.ProgressBar(len(vis_frames))
        for x in vis_frames:
            vid.append_data(x[:, :, ::-1])
            if args.show_progress_bar == 1:
                prog_bar.update()
        vid.close()
        print("\n视频已使用 imageio 保存到:", vidname)

    '''[效率优化] 使用 imageio
    moviepy write_videofile 比 cv2、imageio 都慢, 不是因为它没有使用 gpu 优化, 而是因为每帧 write_frame 时, 
    都需要先 img_array.tobytes() 把帧数据转为二进制格式, 再通过 self.proc.stdin.write 把数据传递给 ffmpeg.
    实际测试中, tobytes() 可以占据 write_videofile 90% 左右的耗时, 并且 tobytes() 是在单线程中调用的, 
    无法利用多核进行并行优化.
    虽然 cv2 是最快的，但 cv2 存在一个问题， 使用 mp4v 格式生成出来的视频无法在浏览器上播放，如果要使用 h264 来生成视频，
    又需要重新编译 opencv-python 库，这样会非常麻烦. 所以，折衷的办法就是使用 imageio 来生成视频。

    时间对比：1分钟视频合成耗时(秒)
    cv2:     3.639
    imageio: 6.115
    moviepy: 11.539
    '''
    write_video_by_imageio(args.out_filename)


@TimeCostDecorator
def main():
    args = parse_args()
    print(args)

    frame_paths, original_frames = frame_extraction(args.video, args.show_progress_bar)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # resize frames to shortside 256
    new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
    resize_frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    w_ratio, h_ratio = new_w / w, new_h / h

    # Get clip_len, frame_interval and calculate center index of each clip
    config = mmcv.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    val_pipeline = config.data.val.pipeline

    sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    # Note that it's 1 based here
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    # Load label_map
    label_map = load_label_map(args.label_map)
    try:
        if config['data']['train']['custom_classes'] is not None:
            label_map = {
                id + 1: label_map[cls]
                for id, cls in enumerate(config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    # Get Human detection results
    '''[效率优化] center_frames 使用已经加载到内存的 original_frames, 以提高运行速度.
    detection_inference 内部调用的 inference_detector 支持 "image files or loaded images".
    '''
    # center_frames = [frame_paths[ind - 1] for ind in timestamps]
    center_frames = [original_frames[ind - 1] for ind in timestamps]
    human_detections = detection_inference(args, center_frames)
    for i in range(len(human_detections)):
        det = human_detections[i]
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

    # Get img_norm_cfg
    img_norm_cfg = config['img_norm_cfg']
    if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
        to_bgr = img_norm_cfg.pop('to_bgr')
        img_norm_cfg['to_rgb'] = to_bgr
    img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
    img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        config['model']['test_cfg']['rcnn']['action_thr'] = .0
    except KeyError:
        pass

    config.model.backbone.pretrained = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))

    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    @TimeCostDecorator
    def SpatioTemporal():
        print('Performing SpatioTemporal Action Detection for each clip')
        predictions = []
        assert len(timestamps) == len(human_detections)
        if args.show_progress_bar == 1:
            prog_bar = mmcv.ProgressBar(len(timestamps))
        for timestamp, proposal in zip(timestamps, human_detections):
            if proposal.shape[0] == 0:
                predictions.append(None)
                continue

            start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
            frame_inds = start_frame + np.arange(0, window_size, frame_interval)
            frame_inds = list(frame_inds - 1)
            imgs = [resize_frames[ind].astype(np.float32) for ind in frame_inds]
            _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
            # THWC -> CTHW -> 1CTHW
            input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
            input_tensor = torch.from_numpy(input_array).to(args.device)

            with torch.no_grad():
                result = model(
                    return_loss=False,
                    img=[input_tensor],
                    img_metas=[[dict(img_shape=(new_h, new_w))]],
                    proposals=[[proposal]])
                result = result[0]
                prediction = []
                # N proposals
                for i in range(proposal.shape[0]):
                    prediction.append([])
                # Perform action score thr
                for i in range(len(result)):
                    if i + 1 not in label_map:
                        continue
                    for j in range(proposal.shape[0]):
                        if result[i][j, 4] > args.action_score_thr:
                            prediction[j].append((label_map[i + 1], result[i][j,
                                                                              4]))
                predictions.append(prediction)
            if args.show_progress_bar == 1:
                prog_bar.update()

        results = []
        for human_detection, prediction in zip(human_detections, predictions):
            results.append(pack_result(human_detection, prediction, new_h, new_w))

        return results

    results = SpatioTemporal()
    resize_frames = None
    gc.collect()

    # 保存结果到 txt 文件
    write_result_txt(args.out_txtname, results)

    # 合成带框图的视频
    if (args.visualize == 1):
        write_visualize_video(args, timestamps, results, original_frames)

    # 删除临时目录
    tmp_frame_dir = osp.dirname(frame_paths[0])
    if os.path.exists(tmp_frame_dir):
        shutil.rmtree(tmp_frame_dir)


'''调用示例：
python detect.py --visualize 1 --show-progress-bar 1 --config ./config/config.py --checkpoint ./checkpoints/best.pth --det-config ./config/det_config.py  --det-checkpoint checkpoints/det_checkpoint.pth   --video ./videos/2.mp4  --out-filename ./results/detb_2.mp4 --out-txtname ./results/detb_2_result.txt   --det-score-thr 0.1 --action-score-thr 0.1 --predict-stepsize 30 --output-stepsize 2  --output-fps 12 --label-map label_map.txt
'''
if __name__ == '__main__':
    main()
    print_all_cost()
