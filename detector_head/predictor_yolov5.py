# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/23 下午7:12
@Author  : Kend
@FileName: yolov5_predictor.py
@Software: PyCharm
@modifier:
"""
import cv2
import numpy as np
import torch
from detector_head.yolov5.yolov_func import non_max_suppression, scale_boxes
from detector_head.yolov5.yolov_model import DetectMultiBackend


class Yolov5nPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights_path = model_path
        self.model = self.load_yolo_model()


    def load_yolo_model(self):
        print(self.weights_path)
        model = DetectMultiBackend(self.weights_path, device=self.device)
        return model

    @staticmethod
    def resize_and_padding(image, target_width=640, target_height=640):
        # 获取原始图像的宽度和高度
        original_height, original_width = image.shape[:2]
        # 计算缩放比例，确保最长边不超过目标框的最长边，也就是说最长边拉满，短边安装比例放缩
        scale = min(target_width / original_width,
                    target_height / original_height)
        # 计算新的宽度和高度
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        # 缩放图像
        resized_image = cv2.resize(image, (new_width, new_height),
                                   interpolation=cv2.INTER_AREA)
        # 创建一个新的空白图像（黑色背景）
        padded_image = np.zeros((target_height, target_width, 3),
                                dtype=np.uint8)
        # 计算居中的位置
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        # 将缩放后的图像粘贴到新的空白图像上
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset +
                                                              new_width] = resized_image

        return padded_image, scale


    def predict(self, image):
        """ 步骤1 给每帧图像顺序处理记录id """
        # 初始化图像信息字典。
        img_info = {"id": 0}  # 初始化图像信息字典。
        # 处理图像路径
        if isinstance(image, str):
            import os.path as osp
            img_info["file_name"] = osp.basename(image)  # 获取图像文件名。
            image = cv2.imread(image)
        else:
            #  如果输入不是一个字符串。
            img_info["file_name"] = None  # 文件名设为 None。

        # 获取图像尺寸
        height, width = image.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = image
        img, scale = self.resize_and_padding(image, target_width=640, target_height=640)
        img_info["ratio"] = scale  # 存储缩放比例。
        im = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # 转换为连续的数组
        im = torch.from_numpy(im).to(self.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # 增加bath size维度

        person_result = []
        # ===========================人体检测 ===================================
        with torch.no_grad():
            person_pre = self.model(im)[0]  # pt模型推理
            person_pre = non_max_suppression(
                person_pre,
                conf_thres=0.5,
                iou_thres=0.45,
                classes=None,
                agnostic=True,
                max_det=100,
                nm=0  # 目标检测设置为0
            )
        # 放缩binding boxes -> image 原图大小
        person_pre[0][:, :4] = scale_boxes(im.shape[2:], person_pre[0][:, :4],
                                           image.shape).round()
        for *xyxy, conf, _cls in person_pre[0]:  # 遍历检测到的人
            # 位置+类别
            person_result.append([
                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), _cls.item(), conf.item(), 0.45]
                )
        return person_result, img_info



if __name__ == '__main__':
    model_path = r'D:\kend\other\yolov5n.pt'
    predictor = Yolov5nPredictor(model_path=model_path)
    img = cv2.imread(r"D:\kend\WorkProject\Hk_Tracker\data\dataset\test_images\frame_0000.jpg")
    re, img_info = predictor.predict(img)
    print(re, "\n")
    img_info["raw_img"] = None
    print(img_info)