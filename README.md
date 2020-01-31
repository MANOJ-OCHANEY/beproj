This repository contains the code for our real-time pose based action recognition.

The approach is based on encoding human poses over a period of time in an image-like data format (Encoded Human Pose Images) that can then be classified using standard CNNs. The entire pipeline is lightweight and runs 20-60 FPS, depending on settings.

# Installation
## Prerequisites
- Python 3.6+
- CUDA (tested with 9.0 and 10.0)
- CUDNN (tested with 7.5)
- PyTorch (tests with 1.0)
- OpenCV with Python bindings

## Setup

```bash
git clone https://github.com/MANOJ-OCHANEY/beproj.git
pip install torchvision
pip install -r req.txt 

cd /data/models
wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/ehpi_v1.pth
wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/yolov3.weights
wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/pose_resnet_50_256x192.pth.tar

```
An example showing the whole pipeline on the webcam can be executed as follows:
```bash
python ehpi_action_recognition/run_ehpi.py
```

## Configuration Options

There are some configuration options available in run_ehpi.py:

- image_size = ImageSize(width=640, height=360): The image size to be used. Higher resolutions usually help Yolo to detect objects.
- camera_number = 0: The webcam id
- fps = 30: FPS which should be used for the input source (webcam or image folder)
- buffer_size = 20: The size of the action buffer, in this project not really used, just the detected humans from frame n-1.
- action_names = [Action.IDLE.name, Action.WALK.name]: The corresponding names to the action class vector outputed by the action recognition network. Need to be updated when you train your own models with different action classes.
- use_action_recognition = True: Turns the action recognition on / off
- use_quick_n_dirty = False: If set to true it deactivates the object recognition completly after a human skeleton has been found. Continues to track this skeleton but won't recognize new humans. Improves the performance by a huge margin.