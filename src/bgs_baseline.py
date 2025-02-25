# %% init

# add morph operation 
import os
import cv2
import numpy as np
from IPython.display import display, clear_output, HTML
from tqdm import tqdm
from PIL import Image
import argparse
from eval import BinaryConfusion
from median_bgs import MedianBGS

confusion = BinaryConfusion(backend="numpy")
parser = argparse.ArgumentParser()
parser.add_argument("--video_id", type=str, default="vid1_")
parser.add_argument("--root_path", type=str, default="../sim")
parser.add_argument("--log_file", type=str, default="results_mog_50_ellipse.csv")
parser.add_argument("--morph", type=int, default=20)
parser.add_argument("--bgs_method", type=str, default="mog", choices=["mog", "knn", "median"])
args = parser.parse_args()
current_video_id = args.video_id
root_path = args.root_path
# wandb.init("LangBGS2", name=str(current_video_id))

videos = list(set(["_".join(i.split("_")[:-1]) for i in os.listdir(os.path.join(root_path, "in"))]))
frames = sorted([i for i in os.listdir(os.path.join(root_path, "in")) if i.startswith(current_video_id + "_0")])
gt_id = current_video_id.split("_")[0].replace("vid","smoke_only")+"_0"
gt_frames = sorted([i for i in os.listdir(os.path.join(root_path, "gt")) if i.startswith(gt_id)])
assert len(frames) > 0, "No frames found, video should be one of " + str(videos)
assert len(gt_frames) > 0, "No gt frames found, video should be one of " + str(videos)
assert len(frames) == len(gt_frames),("Number of frames and gt frames should be equal ", len(frames), len(gt_frames), current_video_id)


if args.bgs_method == "mog":
    bgsub = cv2.createBackgroundSubtractorMOG2(history=30)
elif args.bgs_method == "knn":
    bgsub = cv2.createBackgroundSubtractorKNN(history=30)
elif args.bgs_method == "median":
    bgsub = MedianBGS()
else:
    raise ValueError("Invalid BGS method")
    
    
index = 0
frame_level_pred = []
frame_level_gt = []

for frame in tqdm(frames):
    filename = f'{current_video_id}_{frame.split(".")[0]}'
    img = cv2.imread(os.path.join(root_path, "in", frame))
    bgsub.apply(img)
    if index % 5 != 0:
        index += 1
        continue
    bg = bgsub.getBackgroundImage().astype(float)
    diff = cv2.absdiff(img.astype(float), bg) * 15 #too large cannot see detial, 2 step and clip each time? target value same
    diff = np.clip(diff, 0, 128).astype(np.uint8)
    mask = diff.mean(-1) > 40
    
    # if args.use_morph:
    # opening 3x3 using circle
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
    # closing 50x50
    if args.morph > 0:
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morph, args.morph)))

    
    gt = cv2.imread(os.path.join(root_path, "gt", gt_frames[index]))
    gt_binary = (gt.mean(-1) > 20).astype(np.uint8)
    
    confusion.update(gt_binary, mask)
    frame_level_pred.append(mask.any())
    frame_level_gt.append(gt_binary.any())
    print(confusion.get_f1(), confusion.get_iou(), confusion.get_precision(), confusion.get_recall())

    index += 1
    # wandb.log({"f1": confusion.get_f1(),
    #         "iou": confusion.get_iou(),
    #         "precision": confusion.get_precision(),
    #         "recall": confusion.get_recall(),
    #         "frame": wandb.Image(mask)
    #         })
    
# print(frame_level_pred)
# print(frame_level_gt)
frame_level_acc = (np.array(frame_level_pred) == np.array(frame_level_gt)).mean()
if not os.path.exists(args.log_file):
    with open(args.log_file, "w") as f:
        f.write("video_id,f1,iou,precision,recall,fla,morph,method\n")

with open(args.log_file, "a") as f:
    f.write(f"{current_video_id},{confusion.get_f1()},{confusion.get_iou()},{confusion.get_precision()},{confusion.get_recall()},{frame_level_acc},{args.morph},{args.bgs_method}\n")