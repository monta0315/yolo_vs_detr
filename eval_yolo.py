import argparse
import time  # 追加

import ipywidgets as widgets
import matplotlib.pyplot as plt
import requests
import torch
import torchvision.transforms as T
from IPython.display import clear_output, display
from PIL import Image
from torch import nn
from torchvision.models import resnet50


def main():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")


if __name__ == "__main__":
    main()
