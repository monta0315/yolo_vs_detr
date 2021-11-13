import argparse
import time  # 追加

import cv2  # 追加
import ipywidgets as widgets
import matplotlib.pyplot as plt
import requests
import torch
import torchvision.transforms as T
from IPython.display import clear_output, display
from PIL import Image
from torch import nn
from torchvision.models import resnet50

# COCO classes
CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def get_args_parser():
    parser = argparse.ArgumentParser("Set Some args", add_help=False)
    parser.add_argument(
        "--gpu", default=False, type=bool, help="if use GPU, give --gpu"
    )
    parser.add_argument(
        "--display",
        default=False,
        type=bool,
        help="if display evaled video, give --display",
    )

    return parser


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# 以下関数を修正
def put_rect(cv2_img, prob, boxes):
    colors = COLORS * 100
    output_image = cv2_img
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        xmin = (int)(xmin)
        ymin = (int)(ymin)
        xmax = (int)(xmax)
        ymax = (int)(ymax)
        c[0], c[2] = c[2], c[0]
        c = tuple([(int)(n * 255) for n in c])
        output_image = cv2.rectangle(
            output_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4
        )
        cl = p.argmax()
        text = f"{CLASSES[cl]}: {p[cl]:0.2f}"
        output_image = cv2.rectangle(
            output_image,
            (xmin, ymin - 20),
            (xmin + len(text) * 10, ymin),
            (0, 255, 255),
            -1,
        )
        output_image = cv2.putText(
            output_image,
            text,
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
    return output_image


def use_gpu(video_capture, transform, model, fps, out, display):
    model = model.cuda()
    start = time.time()
    _, frame = video_capture.read()
    frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame_cvt)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).cuda()
    # propagate through the model
    with torch.no_grad():
        outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep].cpu(), im.size)

    # display
    output_image = put_rect(frame, probas[keep], bboxes_scaled)

    # End time
    end = time.time()
    # Time elapsed
    seconds = end - start
    print("time:{:.3f} msec".format(seconds * 1000))

    # Calculate frames per second
    fps = (fps + (1 / seconds)) / 2

    cv2.putText(
        output_image,
        "{:.2f}".format(fps) + " fps",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        3,
    )
    if display:
        cv2.imshow("Video", output_image)

    out.write(output_image)


def use_cpu(video_capture, transform, model, fps, out, display):
    start = time.time()
    _, frame = video_capture.read()
    frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame_cvt)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    with torch.no_grad():
        outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], im.size)

    # display
    output_image = put_rect(frame, probas[keep], bboxes_scaled)

    # End time
    end = time.time()
    # Time elapsed
    seconds = end - start
    print("time:{:.3f} msec".format(seconds * 1000))

    # Calculate frames per second
    fps = (fps + (1 / seconds)) / 2

    cv2.putText(
        output_image,
        "{:.2f}".format(fps) + " fps",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        3,
    )
    if display:
        cv2.imshow("Video", output_image)

    out.write(output_image)


def main(args):
    # standard PyTorch mean-std input image normalization
    transform = T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    if args.gpu:
        model = model.cuda()

    video_capture = cv2.VideoCapture("data/test_india.mp4")

    # 幅と高さを取得
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    # 総フレーム数とフレームレートを取得
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画読み込みの場合
    print("動画のフレームカウント数", frame_count)
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))

    fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter("./result/detr_result.mp4", fmt, frame_rate, size)

    seconds = 0.0
    fps = 0.0

    if args.gpu:
        for i in range(frame_count):
            use_gpu(video_capture, transform, model, fps, out, args.display)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press Q to stop!
                break
    else:
        for i in range(frame_count):
            use_cpu(video_capture, transform, model, fps, out, args.display)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press Q to stop!
                break

    video_capture.release()  # 読み込んだ動画やカメラデバイスを閉じるにはrelease()メソッドを実行する。
    out.release()
    cv2.destroyAllWindows()
    print("everything is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluate Trafic Video Fps by  YOLO", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
