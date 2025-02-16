# %% init
import os
import cv2
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import pylab
import numpy as np
from IPython.display import display, clear_output, HTML
from torchvision.ops import box_convert
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoConfig
import torchvision
import bitsandbytes
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import wandb
import argparse
from boxes import get_valid_boxes

parser = argparse.ArgumentParser()
parser.add_argument("--video_id", type=str, default="vid1_")
parser.add_argument("--root_path", type=str, default="../sim")
parser.add_argument("--log_file", type=str, default="results_mog.csv")
args = parser.parse_args()
current_video_id = args.video_id
root_path = args.root_path
wandb.init("LangBGS2", name=str(current_video_id))

videos = list(set(["_".join(i.split("_")[:-1]) for i in os.listdir(os.path.join(root_path, "in"))]))
frames = sorted([i for i in os.listdir(os.path.join(root_path, "in")) if i.startswith(current_video_id + "_0")])
gt_id = current_video_id.split("_")[0].replace("vid","smoke_only")+"_0"
gt_frames = sorted([i for i in os.listdir(os.path.join(root_path, "gt")) if i.startswith(gt_id)])
assert len(frames) > 0, "No frames found, video should be one of " + str(videos)
assert len(gt_frames) > 0, "No gt frames found, video should be one of " + str(videos)
assert len(frames) == len(gt_frames),("Number of frames and gt frames should be equal ", len(frames), len(gt_frames), current_video_id)

bgsub = cv2.createBackgroundSubtractorMOG2(history=30)
for frame in tqdm(frames):
    filename = f'{current_video_id}_{frame.split(".")[0]}'
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        img = cv2.imread(os.path.join(root_path, "in", frame))
        bgsub.apply(img)
        bg = bgsub.getBackgroundImage().astype(float)
        diff = cv2.absdiff(img.astype(float), bg) * 15 #too large cannot see detial, 2 step and clip each time? target value same
        diff = np.clip(diff, 0, 128).astype(np.uint8)
        pred = diff > 10
        confusion.update(torch.tensor(gt_binary), torch.tensor(mask))
        print(confusion.get_f1(), confusion.get_iou(), confusion.get_precision(), confusion.get_recall())