import cv2
import torchvision.transforms as transforms
import PIL.Image
from torch2trt import TRTModule
import torch
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import json
import trt_pose.coco
OPTIMIZED_MODEL = 'resnet18_trt.pth'
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
import time
WIDTH = 224
HEIGHT = 224
# img_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])
# data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
# t0 = time.time()
# torch.cuda.current_stream().synchronize()
# for i in range(50):
#     y = model_trt(data)
# torch.cuda.current_stream().synchronize()
# t1 = time.time()

# print(50.0 / (t1 - t0))


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )
    return kpoint

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]



with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)
def execute(image,frame):
    color = (0, 255, 0)
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    # draw_objects(image, counts, objects, peaks)
    for i in range(counts[0]):
        keypoints = get_keypoint(objects, i, peaks)
        for j in range(len(keypoints)):
            if keypoints[j][1]:
                x = round(keypoints[j][2] * frame.shape[1] )
                y = round(keypoints[j][1] *frame.shape[0]  )
                cv2.circle(frame, (x, y), 3, color, 2)
                # cv2.putText(image , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                # cv2.circle(image, (x, y), 3, color, 2)
    return frame
    


cap = cv2.VideoCapture("demo.mp4")
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (854,480))
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        framec=cv2.resize(frame,(224,224))
        # cp=frame.coppy()
        print(frame.shape)
        t=time.time()
        frame=execute(framec,frame)
        fps=str(1/(time.time()-t))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
 
        cv2.imshow('Frame',frame)
        out.write(frame)
        
        # print()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break
out.release()

cap.release()
cv2.destroyAllWindows()