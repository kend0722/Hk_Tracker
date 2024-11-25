## 这是一个基于ByteTrack和YOLOv5的多目标跟踪项目

## 项目介绍
本项目是一个基于ByteTrack和YOLOv5的多目标跟踪项目，它使用YOLOv5作为目标检测头，使用ByteTrack作为多目标跟踪算法。为了方便理解我的项目，我把检测头和跟踪算法拆分开了
总所周知YOLOv5是一个流行的目标检测模型，而ByteTrack是一个高效的多目标跟踪算法。

## 项目结构
本项目包含以下文件和目录：
- `data/`：存放数据集和一些单元测试的数据。
- -`detector_head/`：目标检测头，目标支持的有yolov5和yolo11，当然你可以在里面按照你的需求更改
- `predictor.py/`：你的具体目标检测头实现的基类。
- `fastapi`：fastapi，我们考虑采用海葵云的分布式平台去部署，所以我留给一个实现api的目录
- `test`：包涵了一些demo和功能测试，可以只用看demo，因为test目录是我个人的习惯
- `tracking`：ByteTrack算法的实现模块，用于多目标跟踪。
- `byte_tracker.py`：ByteTrack算法的主要实现类，需要多次阅读代码，每次都会有不同的感悟
- `utils`：一些其他的工具包和组件
- `main.py`：主程序，用于调用YOLOv5和ByteTrack进行多目标跟踪。
- `Readme.md`：项目说明文档。
- `requirements.txt`：项目依赖的Python库。

## 致谢
本项目基于以下项目进行了修改和优化：
- [YOLOv5](https://github.com/ultralytics/yolov5)：目标检测模型。
- [ByteTrack](https://github.com/ifzhang/ByteTrack)：多目标跟踪算法。
- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)：YOLOv5的官方实现。
- 特别需要感谢海葵云部门，为我们搭建了分布式集群，能够很好的部署我们的算法

