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

from data.dataset.preprocess import preprocess_image
# from detector_head.yolov5.yolov_func import non_max_suppression, scale_boxes
from detector_head.yolov5.yolov_model import DetectMultiBackend
from track_utils.my_timer import MyTimer
from track_utils.post_process.post_processing import postprocess


class Yolov5nPredictor:



    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights_path = model_path
        self.model = self.load_yolo_model()
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        # self.test_size = (640, 640)
        self.input_size = (640, 640)
        self.fp16 = None


    def load_yolo_model(self):
        # print(self.weights_path)
        model = DetectMultiBackend(self.weights_path, device=self.device)
        return model

    @staticmethod
    def resize_and_padding(image, target_width=1280, target_height=720):
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
        # 处理图像路径
        img_info = {"id": 0}
        if isinstance(image, str):
            import os.path as osp
            img_info["file_name"] = osp.basename(image)
            image = cv2.imread(image)
        else:
            img_info["file_name"] = None

        height, width = image.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = image

        img, ratio = preprocess_image(image, self.input_size, self.rgb_means, self.std)
        print("ratio:", ratio)
        img_info["ratio"] = ratio
        im = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16 is not None:
            im = img.half()  # to FP16
        else:
            im = im.float()  # uint8 to fp16/32
        # # 获取图像尺寸
        # img, scale = self.resize_and_padding(image, target_width=640, target_height=640)
        # img_info["ratio"] = scale
        # im = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # im = np.ascontiguousarray(im)  # 转换为连续的数组
        # im = torch.from_numpy(im).to(self.device)
        # im /= 255  # 0 - 255 to 0.0 - 1.0
        # if len(im.shape) == 3:
        #     im = im[None]  # 增加bath size维度
        # person_result = []
        # ===========================人体检测 ===================================
        with torch.no_grad():
            person_pre = self.model(im)[0]  # pt模型推理
            # self.num_classes, self.confthre, self.nmsthre(1, 0.001, 0.7)
            outputs = postprocess(
                person_pre, 1, 0.01, 0.7
            )
            # print(outputs, "outputs")
            # person_pre = non_max_suppression(
            #     person_pre,
            #     conf_thres=0.5,
            #     iou_thres=0.45,
            #     classes=None,
            #     agnostic=True,
            #     max_det=100,
            #     nm=0  # 目标检测设置为0
            # )
        # 放缩binding boxes -> image 原图大小
        # person_pre[0][:, :4] = scale_boxes(im.shape[2:], person_pre[0][:, :4],
        #                                    image.shape).round()
        # print('person_pre1', person_pre[0])
        # for *xyxy, conf, _cls in person_pre[0]:  # 遍历检测到的人
        #     # 位置+类别
        #     person_result.append([
        #         int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), _cls.item(), conf.item()]
        #         )
        return outputs, img_info


if __name__ == '__main__':
    MyTimer = MyTimer()
    MyTimer.start()
    model_path = r'/home/lyh/work/yolov5s.pt'
    predictor = Yolov5nPredictor(model_path=model_path)
    img_path = r"/home/lyh/work/Hk_Tracker/data/dataset/test_images/frame_0000.jpg"
    output = predictor.predict(img_path)
    print(output)
    duration = MyTimer.stop(average=False)
    print(duration)
