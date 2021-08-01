import argparse
import os
import torch
from trt import *
import glob
import random

import time
import cv2
import numpy as np

from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import get_func_heatmap_to_coord
# from darknet_images import vis_v5 
from queue import  Queue

import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import infer as inf
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
from tracker import pipeline

_input_size=(256,192)
cfg = update_config("../configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml")
# checkpoint 
import math
hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
heatmap_to_coord = get_func_heatmap_to_coord(cfg)


device = torch.device("cuda:0")
norm_type = cfg.LOSS.get('NORM_TYPE', None)
pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
transformation = SimpleTransform(
            pose_dataset, scale_factor=0,
            input_size=_input_size,
            output_size=(64,48),
            rot=0, sigma=2,
            train=False, add_dpg=False, gpu_device=device)
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
print('Loading pose model')
pose_model.load_state_dict(torch.load("../pretrained_models/fast_res50_256x192.pth", map_location=device))
pose_model.to(device)
pose_model.eval()
l_pair = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (17, 11), (17, 12),  # Body
    (11, 13), (12, 14), (13, 15), (14, 16)
]

p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
            (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
            (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                (77, 222, 255), (255, 156, 127),
                (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]


def vis_v5(hm_data, detection,cropped_boxes,frame):
    assert hm_data.dim() == 4
                #pred = hm_data.cpu().data.numpy()
    eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    if hm_data.size()[1] == 136:
        eval_joints = [*range(0,136)]
    elif hm_data.size()[1] == 26:
        eval_joints = [*range(0,26)]
    pose_coords = []
    pose_scores = []
    for i in range(hm_data.shape[0]):
        bbox = cropped_boxes[i].tolist()
        pose_coord, pose_score = heatmap_to_coord(hm_data[i][eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
        pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
        pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
    preds_img = torch.cat(pose_coords)
    preds_scores = torch.cat(pose_scores)

    _result = []
    for k in range(len(detection)):
        # print( float(torch.mean(preds_scores[k])),detection[k][1],float(1.25 * max(preds_scores[k])))
        _result.append(
            {
                'keypoints':preds_img[k],
                'kp_score':preds_scores[k],
                # 'proposal_score': float(torch.mean(preds_scores[k])) + float(detection[k][1]) + float(1.25 * max(preds_scores[k])),
                'box':[ detection[k][0], detection[k][1], detection[k][2]-detection[k][0],detection[k][3]-detection[k][1]] 
            }
        )

    result = {
        'result': _result
    }
    img = frame.copy()
    # print(_result)
    height, width = img.shape[:2]
    for human in result['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        # if kp_num == 17:
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))



        # Draw keypoints
        vis_thres = 0.4
        for n in range(kp_scores.shape[0]):
            # print("shape",kp_scores.shape[0])
            if kp_scores[n] <= vis_thres:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            bg = img.copy()
            if n < len(p_color):
                
                cv2.circle(bg, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
            else:
                cv2.circle(bg, (int(cor_x), int(cor_y)), 1, (255,255,255), 2)
            # Now create a mask of logo and create its inverse mask also
            if n < len(p_color):
                transparency = float(max(0, min(1, kp_scores[n])))
            else:
                transparency = float(max(0, min(1, kp_scores[n]*2)))
            img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
        # Draw limbs
        # for i, (start_p, end_p) in enumerate(l_pair):
        #     if start_p in part_line and end_p in part_line:
        #         start_xy = part_line[start_p]
        #         end_xy = part_line[end_p]
        #         bg = img.copy()

        #         X = (start_xy[0], end_xy[0])
        #         Y = (start_xy[1], end_xy[1])
        #         mX = np.mean(X)
        #         mY = np.mean(Y)
        #         length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
        #         angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
        #         stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
        #         polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), int(stickwidth)), int(angle), 0, 360, 1)
        #         if i < len(line_color):

        #             cv2.fillConvexPoly(bg, polygon, line_color[i])
        #         else:
        #             cv2.line(bg, start_xy, end_xy, (255,255,255), 1)
        #         if n < len(p_color):
        #             transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
        #         else:
        #             transparency = float(max(0, min(1, (kp_scores[start_p] + kp_scores[end_p]))))

        #         #transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
        #         img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    # cv2.imshow("imgaaa",bg)
    return img

def convert2original(image, bbox):
    # print(bbox)
    x, y, w, h =bbox
    _height     = 384
    _width      = 640
    image_h, image_w, __ = image.shape
    # print(image_h, image_w)
    orig_x       = int(x/_width * image_w)
    orig_y       = int(y/_height * image_h)
    orig_width   = int(w/_width * image_w)
    orig_height  = int(h/_height * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def ske(orig_img,detection):
    
    inps = torch.zeros(len(detection), 3, *_input_size)
    
                # print
    cropped_boxes = torch.zeros(len(detection), 4)
    for i, box in enumerate(detection):
        # print(detection[2])
        # print("dang trong day")
        inps[i], cropped_box =transformation.test_transform(orig_img, box)
        cropped_boxes[i] = torch.FloatTensor(cropped_box)
        # print("box",box[2])
        # print("crbox",cropped_boxes[i])
    # print(inps.size())
    hm_j=None
    if len(detection)>0:
        with torch.no_grad():
            t=time.time()
            inps = inps.to(device)
            hm_j = pose_model(inps)
            # print(time.time()-t)
    # i=vis(hm_j, detection,cropped_boxes,orig_img)
    return orig_img, hm_j,cropped_boxes
    

def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height

# class inferThread(threading.Thread):
#     def __init__(self, yolov5_wrapper, cap):
#         threading.Thread.__init__(self)
#         self.yolov5_wrapper = yolov5_wrapper
#         self.cap = cap

#     def run(self):
#         frame_count = 0 # frame counter

#         max_age = 4  # no.of consecutive unmatched detection before 
#                     # a track is deleted

#         min_hits =1  # no. of consecutive matches needed to establish a track

#         tracker_list =[] # list for trackers
#         while cap.isOpened():
#             t0=time.time()
#         # # print("da trong capture")
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             if not ret:
#                 break
#             image_raws,boxs, use_time = self.yolov5_wrapper.infer(frame)
#             # image_raws,frame_count,tracker_list=pipeline(image_raws,frame_count,tracker_list,max_age,min_hits,boxs)
#             image_raws,hm_j,cropped_boxes=ske(frame,boxs)
#             print(tracker_list)
#             if hm_j is not None:
#                 image_raws=vis_v5(hm_j, boxs,cropped_boxes,image_raws)
#             for trk in tracker_list:
#                 if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
#                     # good_tracker_list.append(trk)
#                     x_cv2 = trk.box
#                     import helpers
#                     image_raws= helpers.draw_box_label(image_raws, x_cv2,trk.id) # Draw the bounding boxes on the 
#             print(1/(time.time()-t0))
#             cv2.imshow('Inference', image_raws)  
#             if cv2.waitKey(1)==27:
#                 break
            
        # cv2.waitKey(30)
        # print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))

def video_Capture_Thread(frame_queue, boxes_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        image_raw,result_boxes, usetime=yolov5_wrapper.infer(frame)
        boxes_queue.put(result_boxes)
        frame_queue.put(frame)

    cap.release()

def inference_Thread(frame_queue,boxes_queue, cropped_boxes_queue,hm_queue):
    while cap.isOpened():
        if frame_queue.empty() or boxes_queue.empty():
            continue
        frame=frame_queue.get()
        frame_queue.put(frame)
        boxs=boxes_queue.get()
        boxes_queue.put(boxs)
        if len(boxs)>0:
            image_raws,hm_j,cropped_boxes=ske(frame,boxs)  
            cropped_boxes_queue.put(cropped_boxes)
            hm_queue.put(hm_j)

    cap.release()
def visualize_Thread(frame_queue,boxes_queue, cropped_boxes_queue,hm_queue):
    while cap.isOpened():
        try:
            if frame_queue.empty() or boxes_queue.empty() or cropped_boxes_queue.empty():
                continue
            image_raws=frame_queue.get()
            boxs=boxes_queue.get()
            cropped_boxes=cropped_boxes_queue.get()
            hm_j=hm_queue.get()
            if hm_j is not None:
                image_raws=vis_v5(hm_j, boxs,cropped_boxes,image_raws)
            cv2.imshow("out", image_raws)
            cv2.waitKey(1)
        except:
            pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # load custom plugins
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "build/yolov5s_int8b1.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels

    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')
    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    frame_queue = Queue()
    cropped_boxes_queue = Queue(maxsize = 1)
    boxes_queue = Queue(maxsize = 1)
    hm_queue= Queue(maxsize = 1)
    thread1 = threading.Thread(target=video_Capture_Thread, args=(frame_queue, boxes_queue ))
    thread2 = threading.Thread(target=inference_Thread, args=(frame_queue,boxes_queue, cropped_boxes_queue,hm_queue))
    thread3 = threading.Thread(target=visualize_Thread, args=(frame_queue,boxes_queue, cropped_boxes_queue,hm_queue))
    try:
        t=time.time()
        cap = cv2.VideoCapture("/home/huycq/Downloads/tracking.mp4")
        ret, frame = cap.read()
        thread1.start()
        thread2.start()
        thread3.start()
        
        thread1.join()
        thread2.join()
        thread3.join()
        # print(time.time)
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()
        cap.release()
    # video.release()
        cv2.destroyAllWindows()
