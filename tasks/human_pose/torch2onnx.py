# import onnx
import torch
import trt_pose.models
import json
import trt_pose.coco

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'

model.load_state_dict(torch.load(MODEL_WEIGHTS))
filepath='model.onnx'
WIDTH = 224
HEIGHT = 224

data = torch.rand((1, 3, HEIGHT, WIDTH)).cuda()
torch.onnx.export(model,data,filepath)
