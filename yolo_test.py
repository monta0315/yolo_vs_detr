import ipywidgets as widgets
import matplotlib.pyplot as plt
import requests
import torch
import torchvision.transforms as T
from IPython.display import clear_output, display
from PIL import Image
from torch import nn
from torchvision.models import resnet50

#%config InlineBackend.figure_format = 'retina'


torch.set_grad_enabled(False)

import argparse
import time  # 追加

import cv2  # 追加


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
    parser.add_argument("--imgsz", default=640, type=int)

    return parser


def use_gpu(video_capture, model, fps, out, display):
    model = model.cuda()
    start = time.time()
    _, frame = video_capture.read()
    frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame_cvt)

    # propagate through the model
    with torch.no_grad():
        outputs = model(im)
    # display
    output_image = outputs.imgs[0]

    # End time
    end = time.time()
    # Time elapsed
    seconds = end - start

    # Calculate frames per second
    fps = (fps + (1 / seconds)) / 2

    print(
        "time:{:.3f} msec".format(seconds * 1000) + "  " + "{:.2f}".format(fps) + " fps"
    )

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


def use_cpu(video_capture, model, fps, out, display):
    start = time.time()
    _, frame = video_capture.read()
    frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame_cvt)

    # propagate through the model
    with torch.no_grad():
        outputs = model(im)

    outputs.render()
    # display
    output_image = outputs.imgs[0]

    # End time
    end = time.time()
    # Time elapsed
    seconds = end - start

    # Calculate frames per second
    fps = (fps + (1 / seconds)) / 2

    print(
        "time:{:.3f} msec".format(seconds * 1000) + "  " + "{:.2f}".format(fps) + " fps"
    )

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
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    video_capture = cv2.VideoCapture("data/test_india.mp4")  # 動画読み込み

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
            use_gpu(video_capture, model, fps, out, args.display)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press Q to stop!
                break
    else:
        for i in range(frame_count):
            use_cpu(video_capture, model, fps, out, args.display)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press Q to stop!
                break

    video_capture.release()  # 読み込んだ動画やカメラデバイスを閉じるにはrelease()メソッドを実行する。
    out.release()
    cv2.destroyAllWindows()
    print("everything is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluate Trafic Video Fps by DETR", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
