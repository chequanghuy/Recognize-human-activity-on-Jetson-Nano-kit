import torch2trt
import torch
import trt_pose.models
import json
import trt_pose.coco
WIDTH = 224
HEIGHT = 224
MODEL_WEIGHTS = 'resnet18.pth'
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(
    num_parts, 2 * num_links).cuda().eval()
model.load_state_dict(torch.load(MODEL_WEIGHTS))
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1 <<25)
OPTIMIZED_MODEL = 'resnet18_trt.pth'

torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
