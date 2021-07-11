# Recognize human activity on Jetson Nano kit



## Getting Started


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

### Step 3 - Run code

1. cd code/tasks/human_pose.  
2. python3 run.py
