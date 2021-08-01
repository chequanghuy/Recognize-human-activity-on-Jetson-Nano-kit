# from alphapose.utils import detector
import os
import sys
from threading import Thread
from queue import Queue
from trt import *
import cv2
import numpy as np

import torch
import torch.multiprocessing as mp
import pose_mp
from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder

class DetectionLoader():
    def __init__(self, input_source, detector, cfg, opt,  batchSize=1, queueSize=128):
        self.cfg = cfg
        self.opt = opt
        # self.mode = mode
        self.device = torch.device("cuda:0")

        # elif mode == 'video':
        stream = cv2.VideoCapture(input_source)
        assert stream.isOpened(), 'Cannot capture source'
        self.path = input_source
        self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
        self.fps = stream.get(cv2.CAP_PROP_FPS)
        self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.videoinfo = {'fourcc': self.fourcc, 'fps': self.fps, 'frameSize': self.frameSize}
        stream.release()

        self.detector = detector
        self.batchSize = batchSize
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        self._input_size = (256,192)
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)

        # initialize the queue used to store data
        """
        image_queue: the buffer storing pre-processed images for object detection
        det_queue: the buffer storing human detection results
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """
        # if opt.sp:
        self._stopped = False
        self.image_queue = Queue(maxsize=queueSize)
        self.det_queue = Queue(maxsize=10 * queueSize)
        self.pose_queue = Queue(maxsize=10 * queueSize)
        

    def start_worker(self, target):
        # if self.opt.sp:
        p = Thread(target=target, args=())
        
        # p.daemon = True
        p.start()
        return p

    def start(self):

        # elif self.mode == 'video':
        image_preprocess_worker = self.start_worker(self.frame_preprocess)
        # start a thread to detect human in images
        image_detection_worker = self.start_worker(self.image_detection)
        # start a thread to post process cropped human image for pose estimation
        image_postprocess_worker = self.start_worker(self.image_postprocess)

        return [image_preprocess_worker, image_detection_worker, image_postprocess_worker]

    def stop(self):
        # clear queues
        self.clear_queues()

    def terminate(self):
        # if self.opt.sp:
        self._stopped = True
        # else:
        #     self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.image_queue)
        self.clear(self.det_queue)
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()


    def frame_preprocess(self):
        stream = cv2.VideoCapture(self.path)
        assert stream.isOpened(), 'Cannot capture source'

        for i in range(self.datalen):
            # print("dang 1")
            # for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
            (grabbed, frame) = stream.read()
            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed or self.stopped:
                # put the rest pre-processed data to the queue
                
                self.wait_and_put(self.image_queue, (None, None, None))
                print('===========================> This video get ' + str(k) + ' frames in total.')
                sys.stdout.flush()
                stream.release()
                return

            image_raw,result_boxes, usetime=self.detector.infer(frame)
            self.wait_and_put(self.image_queue, (image_raw,result_boxes, usetime))
            # im_dim_list_ = im_dim_list

        self.wait_and_put(self.image_queue, (image_raw,result_boxes, usetime))
        stream.release()

    def image_detection(self):
        for i in range(self.datalen):
            # print("dang 2")
            image_raw,result_boxes, usetime = self.wait_and_get(self.image_queue)
            if image_raw is None or self.stopped:
                self.wait_and_put(self.det_queue, (None, None, None, None))
                return

            image_raws,hm_j,cropped_boxes=pose_mp.ske(image_raw,result_boxes)
            self.wait_and_put(self.det_queue, (hm_j, result_boxes,cropped_boxes,image_raws))
            

    def image_postprocess(self):
        for i in range(self.datalen):
            # print("dang 3")
            hm_j, boxs,cropped_boxes,image_raws= self.wait_and_get(self.det_queue)
            if hm_j is not None:
                image_raws=pose_mp.vis_v5(hm_j, boxs,cropped_boxes,image_raws)
            print(image_raws.shape)
            self.wait_and_put(self.pose_queue, (image_raws,0))
            # else:


    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        # if self.opt.sp:
        return self._stopped
        # else:
        #     return self._stopped.value

    @property
    def length(self):
        return self.datalen

        # n += 1
        
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
    from alphapose.utils.config import update_config
    cfg = update_config("../configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml")
    det_loader = DetectionLoader("/home/huycq/Downloads/tracking.mp4", yolov5_wrapper, cfg, 0, batchSize=1, queueSize=128)
    data_len = det_loader.length
    from tqdm import tqdm
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
    det_worker = det_loader.start()
    try:
        for i in im_names_desc:
            t=time.time()
            (img,_) = det_loader.read()
            print(1/(time.time()-t))
            cv2.imshow("out", img)
            cv2.waitKey(1)
        
        # print(time.time)
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()
        # cap.release()
    # video.release()
        cv2.destroyAllWindows()
    