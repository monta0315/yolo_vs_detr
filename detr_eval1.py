import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

import cv2   #追加
import time  #追加

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

#以下関数を修正
def put_rect(cv2_img, prob, boxes):
    colors = COLORS * 100
    output_image = cv2_img
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        xmin = (int)(xmin)
        ymin = (int)(ymin)
        xmax = (int)(xmax)
        ymax = (int)(ymax)
        c[0],c[2]=c[2],c[0]
        c = tuple([(int)(n*255) for n in c])
        output_image = cv2.rectangle(output_image,(xmin,ymin),(xmax,ymax),(0,0,255), 4)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        output_image = cv2.rectangle(output_image,(xmin,ymin-20),(xmin+len(text)*10,ymin),(0,255,255),-1)
        output_image = cv2.putText(output_image,text,(xmin,ymin-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
    return output_image

    #model load
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

model = model.cuda()   #GPUを使用する場合はこちら

#video_capture = cv2.VideoCapture(0)               #USBカメラ入力
video_capture = cv2.VideoCapture("/data/test_india.mp4")#動画読み込み

# 幅と高さを取得
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

#総フレーム数とフレームレートを取得
frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))#動画読み込みの場合
frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))

fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('./result.mp4', fmt, frame_rate, size)

seconds = 0.0
fps = 0.0

#while True:                    # USBカメラ入力の場合
for i in range(frame_count):# 動画読み込みの場合

    start = time.time()
    _, frame = video_capture.read()
    frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im =Image.fromarray(frame_cvt)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).cuda()#GPUを使用する場合
    #img = transform(im).unsqueeze(0)

    # propagate through the model
    with torch.no_grad():
        outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

        # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), im.size)#GPU
    #bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    #display
    output_image = put_rect(frame, probas[keep], bboxes_scaled)

    # End time
    end = time.time()
    # Time elapsed
    seconds = (end - start)
    print("time:{:.3f} msec".format(seconds*1000) )

    # Calculate frames per second
    fps  = ( fps + (1/seconds) ) / 2

    cv2.putText(output_image,'{:.2f}'.format(fps)+' fps',(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),3)
    cv2.imshow('Video', output_image)
    out.write(output_image)

    # Press Q to stop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
out.release()
cv2.destroyAllWindows()
print("everything is done")