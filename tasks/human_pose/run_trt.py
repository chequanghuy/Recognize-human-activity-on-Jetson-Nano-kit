from __future__ import print_function

# import os
# import argparse
import cv2
# import torch
import tensorrt as trt
import common
import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
import time
# import scipy.special
import torchvision.transforms as transforms
from PIL import Image
EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
img_transforms = transforms.Compose([
    
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
def load_engine(trt_file_path, verbose=False):
    """Build a TensorRT engine from a TRT file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    print('Loading TRT file from path {}...'.format(trt_file_path))
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine
# mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
# std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
# device = torch.device('cuda')
# def preprocess(image):
#     global device
#     device = torch.device('cuda')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = Image.fromarray(image)
#     image = transforms.functional.to_tensor(image).to(device)
#     image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
def main():
    engine_file_path = 'model_fp16.trt'
    engine = load_engine(engine_file_path)

    h_inputs, h_outputs, bindings, stream = common.allocate_buffers(engine)


    cap =  cv2.VideoCapture("abc.mp4")
    with engine.create_execution_context() as context:
        while True:
            # _,frame = cap.read()
            frame=cv2.imread("Dance.jpg")
            t1 = time.time()
            frame=cv2.resize(frame,(224,224))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img_transforms(img).numpy()
            # img=preprocess(frame)
            h_inputs[0].host = img
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs, stream=stream)
            t4 = time.time()
            

            out_j = trt_outputs[0]
            print(out_j.shape)
            print(1/(t4-t1))
            cv2.imshow("OUTPUT", frame)
            cv2.waitKey(1)
if __name__ == '__main__':
    main()