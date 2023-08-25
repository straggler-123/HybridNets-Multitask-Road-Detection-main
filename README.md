# HybridNets-Multitask-Road-Detection-main
基于华为Atlas 200I DK A2开发者套件的环境感知方法
![image](https://github.com/straggler-123/HybridNets-Multitask-Road-Detection-main/assets/93413016/676b476a-44aa-4a22-b8f0-154a8d090306)
#快速开始
配置环境：
```
pip install -r requirements.txt
```
使用ONNX_cpu推理图像：
```
python onnx_image.py
```
使用om权重检测单张图像：
```
python om_image.py
```
使用om权重检测视频：
```
python om_video.py
```
使用om权重调用相机进行检测（usb相机）：
```
python om_camera.py
```
#检测效果
![demo_test](https://github.com/straggler-123/HybridNets-Multitask-Road-Detection-main/assets/93413016/f0c431bc-713f-4871-b1c0-93049146b118)
![8_test](https://github.com/straggler-123/HybridNets-Multitask-Road-Detection-main/assets/93413016/8ba85d72-1176-438e-bb88-c32904fae495)
![12_test](https://github.com/straggler-123/HybridNets-Multitask-Road-Detection-main/assets/93413016/0d9c53a8-fe83-4790-8af2-c58d31f256b9)
![21_test](https://github.com/straggler-123/HybridNets-Multitask-Road-Detection-main/assets/93413016/ecc63190-a1a1-4dd7-9e0a-5eefe0947e62)
