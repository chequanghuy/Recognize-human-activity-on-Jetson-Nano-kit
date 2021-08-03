import argparse
import os
import torch
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import get_func_heatmap_to_coord
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
def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="/home/huycq/AlphaPose/examples/demo/",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=3, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov4-tiny.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./cfg/yolov4-tiny.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))
# def convert2relative(bbox):
#     print("daaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
#     """
#     YOLO format use relative coordinates for annotation
#     """
#     x, y, w, h  = bbox
#     _height     = 416
#     _width      = 416
#     return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    # print(bbox)
    x, y, w, h =bbox
    _height     = 416
    _width      = 416
    image_h, image_w, __ = image.shape
    # print(image_h, image_w)
    orig_x       = int(x/_width * image_w)
    orig_y       = int(y/_height * image_h)
    orig_width   = int(w/_width * image_w)
    orig_height  = int(h/_height * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted
def vis(hm_data, detection,cropped_boxes,frame):
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
                'proposal_score': float(torch.mean(preds_scores[k])) + float(detection[k][1]) + float(1.25 * max(preds_scores[k])),
                'box':[ detection[k][2][0], detection[k][2][1], detection[k][2][2]-detection[k][2][0],detection[k][2][3]-detection[k][2][1]] 
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
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), int(stickwidth)), int(angle), 0, 360, 1)
                if i < len(line_color):

                    cv2.fillConvexPoly(bg, polygon, line_color[i])
                else:
                    cv2.line(bg, start_xy, end_xy, (255,255,255), 1)
                if n < len(p_color):
                    transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
                else:
                    transparency = float(max(0, min(1, (kp_scores[start_p] + kp_scores[end_p]))))

                #transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    # cv2.imshow("imgaaa",bg)
    return img
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
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), int(stickwidth)), int(angle), 0, 360, 1)
                if i < len(line_color):

                    cv2.fillConvexPoly(bg, polygon, line_color[i])
                else:
                    cv2.line(bg, start_xy, end_xy, (255,255,255), 1)
                if n < len(p_color):
                    transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
                else:
                    transparency = float(max(0, min(1, (kp_scores[start_p] + kp_scores[end_p]))))

                #transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    # cv2.imshow("imgaaa",bg)
    return img
def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    print(width,height)
    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (416, 416),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)
def ske(orig_img,detection):
    
    inps = torch.zeros(len(detection), 3, *_input_size)
                # print
    cropped_boxes = torch.zeros(len(detection), 4)
    for i, box in enumerate(detection):
        # print(detection[2])
        # print("dang trong day")
        inps[i], cropped_box =transformation.test_transform_tiny(orig_img, box[2])
        cropped_boxes[i] = torch.FloatTensor(cropped_box)
        # print("box",box[2])
        # print("crbox",cropped_boxes[i])
    # print(inps.size())
    with torch.no_grad():
        t=time.time()
        inps = inps.to(device)
        hm_j = pose_model(inps)
        print(time.time()-t)
    # i=vis(hm_j, detection,cropped_boxes,orig_img)
    return orig_img,hm_j,cropped_boxes
    

def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    t1=time.time()
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(images, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    print("pre-pro",time.time()-t1)
    t2=time.time()
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    print("detect",time.time()-t2)
    t3=time.time()
    detections_adjusted=[]
    for label, confidence, bbox in detections:
        print(bbox)
        bbox_adjusted = convert2original(image, bbox)
        print(bbox_adjusted)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))
    # image = darknet.draw_boxes(detections_adjusted, image, class_colors)
    image,hm_j,cropped_boxes=ske(image,detections_adjusted)
    print("pose",time.time()-t3)
    t4=time.time()
    image=vis(hm_j, detections,cropped_boxes,image)
    print("post-pro",time.time()-t4)
    return image


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/adog.jpg', 'data/aperson.jpg', 'data/aeagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    # print(detections)


def main():
    batch_detection_example()
    # args = parser()
    # check_arguments_errors(args)

    # random.seed(3)  # deterministic bbox colors
    # network, class_names, class_colors = darknet.load_network(
    #     args.config_file,
    #     args.data_file,
    #     args.weights,
    #     batch_size=args.batch_size
    # )

    # images = load_images(args.input)

    # index = 0
    # while True:
    #     # loop asking for new image paths if no list is given
    #     if args.input:
    #         if index >= len(images):
    #             index=0
    #         image_name = images[index]
    #     else:
    #         image_name = input("Enter Image Path: ")
    #     prev_time = time.time()
    #     image= image_detection(
    #         image_name, network, class_names, class_colors, args.thresh
    #         )
    #     # if args.save_labels:
    #     #     save_annotations(image_name, image, detections, class_names)
    #     # darknet.print_detections(detections, args.ext_output)
    #     fps = int(1/(time.time() - prev_time))
    #     print("FPS: {}".format(fps))
    #     # if not args.dont_show:
    #     cv2.imshow('Inference', image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     index += 1


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
