# Recognize human activity on Jetson Nano kit

## Getting Started On Laptop


### Step 1 - Install Dependencies

1. Install Cuda 10.2, CuDNN 7.6.5, TensorRT 7.0



2. Install packages

    ```python
    conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
    python -m pip install cython
    python -m pip install pycuda
    sudo apt-get install libyaml-dev
    ```
### Step 2 - Install 

```python
python setup.py build develop
```
### Step 3 - Download model fast_res50_256x192.pth, yolov5s_int8b1.engine

https://drive.google.com/drive/u/1/folders/1z52Ld_XJlsFq9tFokryLm5de5A0jbJkV
1. coppy model fast_res50_256x192.pth to /Recognize-human-activity-on-Jetson-Nano-kit/AlphaPose+yolov5/pretrained_models
2. coppy model yolov5s_int8b1.engine to /Recognize-human-activity-on-Jetson-Nano-kit/AlphaPose+yolov5/yolov5/build
### Step 3 - Run code

1. cd /Recognize-human-activity-on-Jetson-Nano-kit/AlphaPose/yolov5
2. python pose.py
3. python pose_mp.py (for multiple threads)


## Getting Started On Jetson Nano

### Step 1 - Install Dependencies
1. Install PyTorch and Torchvision.  To do this on NVIDIA Jetson, we recommend following [this guide](https://forums.developer.nvidia.com/t/72048)



2. Install packages

    ```python
    sudo pip3 install tqdm cython pycocotools
    sudo apt-get install python3-matplotlib
    ```

### Step 2 - Install 

```python
cd code
sudo python3 setup.py install
```
### Step 3 - Download model resnet18_trt.pth

https://drive.google.com/drive/u/1/folders/1z52Ld_XJlsFq9tFokryLm5de5A0jbJkV

### Step 3 - Run code

1. cd code/tasks/human_pose.  
2. coppy model resnet18_trt.pth to code/tasks/human_pose.  
3. python3 run.py




