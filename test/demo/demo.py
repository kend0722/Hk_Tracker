#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:44
@Description: demo - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""
import sys
sys.path.append('/home/lyh/work/Hk_Tracker')
import argparse
import os.path as osp
import cv2
import time
import os
import torch
from loguru import logger
from tracking.byte_tracker import BYTETracker
from track_utils.model_utils import fuse_model
from track_utils.my_timer import MyTimer
from visualization.visualize import plot_tracking
from detector_head.predictor_yolov5 import Yolov5nPredictor
from visualization.build import get_exp

# 检测图像的类型
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


"""图像推理的演示方法 image_demo，后续还有视频的"""
def image_demo(predictor, args):
    """
    图像推理的演示方法
    predictor: 预测器对象，用于图像推理. outputs, img_info
    vis_folder: 可视化结果保存的文件夹
    current_time: 当前时间，用于生成唯一文件名
    args: 命令行参数
    Returns:
    """
    # 获取图像文件列表, 处理图像路径
    if osp.isdir(args.path):
        files = get_image_list(args.path)   # 获取目录中的所有图像文件列表。
    else:
        # 如果输入路径是一个文件。
        files = [args.path]  # 将文件路径添加到列表中。
    files.sort()    # 对文件列表进行排序。按照时间的名字排序
    # 初始化追踪器。# TODO 追踪算法
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = MyTimer()  # 初始化计时器对象。
    results = []      # 初始化结果列表。

    # 遍历图像文件， 遍历图像文件列表，从1开始编号。
    for frame_id, img_path in enumerate(files, 1):
        # 使用预测器对图像进行推理，获取输出和图像信息。
        outputs, img_info = predictor.predict(img_path)  # TODO 替换为yolov5， 而且这里是可以直接推理出来所有的图像结果的
        # print(outputs)      # 输出这一帧率中所有的目标位置信息和置信度 这里还是tensor数据类型
        if outputs[0] is not None:
            # print("outputs[0]", outputs[0])
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']])
            # TODO  更新跟踪器
            # online_targets = tracker.update(outputs, img_info) #
            online_tlwhs = []  # 初始化目标边界框列表。
            online_ids = []  # 初始化目标边界框列表。
            online_scores = []  # 初始化目标得分列表。
            for t in online_targets:   # 初始化目标得分列表。
                tlwh = t.tlwh   # 初始化目标得分列表。
                tid = t.track_id    # 获取目标的ID。
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh  # 判断目标是否垂直。
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:  # 如果目标面积大于最小面积且不是垂直目标。
                    online_tlwhs.append(tlwh)   # 添加目标边界框。
                    online_ids.append(tid)  # 添加目标ID。
                    online_scores.append(t.score)   # 添加目标得分。
                    # 记录结果。
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.stop()  # 结束计时。
            # 绘制跟踪结果。后续可删除
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:   # 如果推理结果为空。
            timer.stop()  # 结束计时。
            online_im = img_info['raw_img']  # 使用原始图像。因为需要可视化结果

        """-------------------------------------------------------------------------------------"""
    #     #  如果命令行参数中设置了保存结果。
    #     if args.save_result:
    #         timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    #         # 构建保存目录路径。
    #         save_folder = osp.join(vis_folder, timestamp)
    #         os.makedirs(save_folder, exist_ok=True)
    #         # 保存绘制了跟踪结果的图像。
    #         cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)
    #     # 每处理20帧打印一次进度信息。
    #     if frame_id % 20 == 0:
    #         logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
    #     # 如果用户按下ESC键或Q键
    #     ch = cv2.waitKey(0)
    #     if ch == 27 or ch == ord("q") or ch == ord("Q"):
    #         break
    #
    # # 保存最终结果
    # if args.save_result:    # 如果命令行参数中设置了保存结果。
    #     res_file = osp.join(vis_folder, f"{timestamp}.txt")  # 如果命令行参数中设置了保存结果。
    #     with open(res_file, 'w') as f:
    #         f.writelines(results)
    #     logger.info(f"save results to {res_file}")

def imageflow_demo(predictor, vis_folder, current_time, args):
    print(args, "222")
    # --------------- no pro
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    tracker = BYTETracker(args, frame_rate=30)
    timer = MyTimer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.predict(frame)
            if outputs[0] is not None:
                # logger.info(f"检测头推理的结果：{outputs[0]}")   # 二维张量， 每一行都是7个数
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']])
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.stop()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.stop()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


"""获取路径中的所有图像"""
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


"""将多目标跟踪结果写入文件的函数。
这个函数接受一个文件名和结果数据，然后按照指定的格式将结果写入文件。
"""
def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def video2images(video_path, output_folder, image_format='jpg'):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件 {video_path}")
    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            # 构建图像文件名
            image_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.{image_format}")
            # 保存图像文件
            cv2.imwrite(image_filename, frame)
        frame_count += 1
    # 释放视频捕获对象
    cap.release()
    print(f"转换完成，共生成 {frame_count} 张图像。")


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="/home/lyh/work/Hk_Tracker/data/videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument("-c", "--ckpt", default='/home/lyh/work/yolov5s.pt', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser



def main(args):
    print(args.model_path, "11111")
    predictor = Yolov5nPredictor(model_path=args.model_path)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, args.save_result, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, args.save_result, current_time, args)


class Args:
    demo= "video"
    path = "/home/lyh/work/Hk_Tracker/data/videos/palace.mp4"
    save_result= "/home/lyh/work/Hk_Tracker/visualization/demo_output"
    model_path= "/home/lyh/work/person_v4.pt"
    fps= 30
    track_thresh= 0.5
    track_buffer= 30
    match_thresh= 0.8
    min_box_area= 32
    aspect_ratio_thresh = 1.6
    # args = {
    #     "demo": "video",
    #     "save_result": "/home/lyh/work/Hk_Tracker/visualization/demo_output",
    #     "model_path": "/home/lyh/work/lkkk/person_v4.pt",
    #     "fps": 30,
    #     "track_thresh": 0.5,
    #     "track_buffer": 30,
    #     "match_thresh": 0.8,
    #     "min_box_area": 32,
    #     "aspect_ratio_thresh": 1.6
    # }


if __name__ == "__main__":
    # args = make_parser().parse_args()
    args = Args()
    # logger.info(f"args:{args}")
    """
    args:Namespace(demo='video', experiment_name=None, name=None, path='/home/lyh/work/Hk_Tracker/data/videos/palace.mp4', camid=0, save_result=True, exp_file=None, ckpt='/home/lyh/work/yolov5s.pt', device='gpu', conf=None, nms=None, tsize=None, fps=30, fp16=True, fuse=True,
    trt=False, track_thresh=0.5, track_buffer=30, match_thresh=0.8, aspect_ratio_thresh=1.6, min_box_area=10, mot20=False)
    """
    # print(args.save_result)
    main(args)

    # python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py  --fp16 --fuse --save_result





# def main(args):
#     model_path = r'/home/lyh/work/yolov5s.pt'
#     predictor = Yolov5nPredictor(model_path=model_path)
#     # img = cv2.imread(r"/home/lyh/work/Hk_Tracker/data/dataset/test_images/frame_0000.jpg")
#     # re = predictor.predict(img)
#     # vis_folder = r"/home/lyh/work/Hk_Tracker/visualization/demo_output"
#     # current_time = time.localtime()
#     image_demo(predictor,args)


# 示例用法
# if __name__ == "__main__":
#     """
#     [args.save_result, args.path, args.fps, args.track_thresh, args.track_buffer, args.match_thresh, args.mot20, args.min_box_area, args.aspect_ratio_thresh  # 判断目标是否垂直。]
#     """
#
#
#     # # 解析命令行参数
#     # def make_parser():
#     #     parser = argparse.ArgumentParser("ByteTrack Demo!")
#     #     parser.add_argument("-expn", "--experiment-name", type=str, default=None)
#     #     parser.add_argument("-n", "--name", type=str, default=None, help="model name")
#     #     parser.add_argument("--path", default=r"/home/lyh/work/Hk_Tracker/data/dataset/test_images",
#     #                         help="path to "
#     #                              "images or video")
#     #     parser.add_argument("--save_result",default="/home/lyh/work/Hk_Tracker/visualization/demo_output", action="store_true", help="whether to save the inference result of "
#     #                                                                    "image/video", )
#     #     # exp file
#     #     parser.add_argument("-f", "--exp_file", default=None, type=str,
#     #                         help="pls input your expriment description file", )
#     #     parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
#     #     parser.add_argument("--device", default="cpu", type=str,
#     #                         help="device to run our model, can either be cpu or gpu", )
#     #     parser.add_argument("--conf", default=None, type=float, help="test conf")
#     #     parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
#     #     parser.add_argument("--tsize", default=None, type=int, help="test img size")
#     #     parser.add_argument("--fps", default=5, type=int, help="frame rate (fps)")
#     #     parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision "
#     #                                                                                         "evaluating.", )
#     #     parser.add_argument("--fuse", dest="fuse", default=False, action="store_true",
#     #                         help="Fuse conv and bn for testing.", )
#     #     parser.add_argument("--trt", dest="trt", default=False, action="store_true",
#     #                         help="Using TensorRT model for testing.",
#     #                         )
#     #     # tracking args
#     #     parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
#     #     parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#     #     parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
#     #     parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
#     #                         help="threshold for filtering out boxes of "
#     #                              "which aspect ratio are above the given value."
#     #                         )
#     #     parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
#     #     parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
#     #     return parser
#
#     args = make_parser().parse_args()
#     print(args)
#     main(args)
