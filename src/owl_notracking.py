import os
import cv2
from transformers import Owlv2Processor, Owlv2ForObjectDetection

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--video_id", type=int, default=1237)
args = parser.parse_args()

# current_video_id = 1237
current_video_id = args.video_id




import wandb
# root_path = "sim/in/"
root_path = f"/home/wg25r/original_frames/MOV_{current_video_id}.mp4/"
gt_path = "/home/wg25r/fastdata/gasvid/train/masks/"




filename_map = {}
video_with_gt = []
for i in os.listdir(gt_path):
    gt_name = i
    pieces = gt_name.split("_")
    video_id = pieces[1]
    frame_id = int(pieces[-1].split(".")[0]) - 1
    query_name = f"{video_id}_{frame_id:06d}"
    filename_map[query_name] = gt_name
    video_with_gt.append(video_id)

print(set(video_with_gt))
if str(current_video_id) not in set(video_with_gt):
    print("\033[91mNo GT for this video\033[0m")
    exit()
    
wandb.init("LangBGS2", name=str(current_video_id))

# print head of the map
for i in list(filename_map.items())[:10]:
    print(i)


videos = list(set(["_".join(i.split("_")[:-1]) for i in os.listdir(root_path)]))
# print(videos)
frames = sorted([i for i in os.listdir(root_path) if i.startswith("")])
assert len(frames) > 0, "No frames found, video should be one of " + str(videos)
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



sam2_checkpoint = ".sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))

# use sam to do tracking
# use reverse? cannot include all list
# pre encode text
texts = ["white steam", "white human, car, bird, bike, and other objects"]
# use the other network for proposal and classification
# texts = ["steam plume, semi-transparent gas, smoke", "white human shadow", "white birds shadow",  "white cars shadow"]
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
config = AutoConfig.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble",
                                                quantization_config={"load_in_4bit": True, 
                                                                     "bnb_4bit_compute_dtype":torch.float16},
                                                config=config)#.to("cuda") make it not quantized actually hurt the performance
# model = model.to("cuda")
img0 = cv2.imread(root_path + frames[0])
history_state = []
bgsub = cv2.createBackgroundSubtractorMOG2(history=30)
video_writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (img0.shape[1] * 2, img0.shape[0] * 2))
query_images = Image.open("prompt.png").convert("RGB")
# pylab.imshow(query_images)
# pylab.show(block=False)

prev_masks = None

def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))


# inference_state = video_predictor.init_state(video_path=root_path)
# video_predictor.reset_state(inference_state)
past_boxes = [] 
# record boxes for past 10 frames. For each new frame, 
# if there is any boxes matched (>t_iou) for 8/10 past frames, 
# then it is a valid box. Send it to sam for tracking and segmentation
if_init = False
out_obj_ids = None
index = 0
valid_boxes = []
pylab.figure(figsize=(15, 10))
masks = torch.zeros((1, 1, 1, 1), device="cuda")
from eval import BinaryConfusion

confusion = BinaryConfusion()

for frame in tqdm(frames):
    filename = f'{current_video_id}_{frame.split(".")[0]}'
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        img = cv2.imread(root_path + frame)
        bgsub.apply(img)
        bg = bgsub.getBackgroundImage().astype(float)
        diff = cv2.absdiff(img.astype(float), bg) * 15 #too large cannot see detial, 2 step and clip each time? target value same
        diff = np.clip(diff, 0, 128).astype(np.uint8)
        if index % 5 != 0:
            index += 1
            continue
        
        diff = Image.fromarray(diff)
        inputs = processor(text=texts, images=diff, return_tensors="pt", padding="longest").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.Tensor([diff.size[::-1]])

        frame = np.array(diff)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.06)

        i = 0
        text = texts[i]
        boxes, logits, phrases = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        positive = boxes[phrases == 0]
        positive_logits = logits[phrases == 0]
        indices = torchvision.ops.nms(positive, positive_logits, 0.3)

        valid_boxes = []    
        for current_box in boxes[indices]:
            matched_frames = 0
            
            for past_frame_boxes in past_boxes[max(-10, -len(past_boxes)):]: 
                ious = torchvision.ops.box_iou(current_box.unsqueeze(0), past_frame_boxes)
                abs_diff = torch.abs(current_box - past_frame_boxes)
                if (ious > 0.3).any() or (abs_diff < 40).all():
                    matched_frames += 1
                    
            if matched_frames >= 1:
                valid_boxes.append(current_box)
        valid_boxes = torch.stack(valid_boxes) if valid_boxes else torch.empty((0, 4), device=positive.device)
        
        if len(valid_boxes) > 0:        
            predictor.set_image(frame)
            
            masks, _, _ = predictor.predict(box=valid_boxes, multimask_output=False)

            # forgot somthing here sum
            mask = masks.sum(0) > 0
            
            if len(mask.shape) == 3:
                mask = mask[0]
            
            frame[mask,0] = 255
                
        if positive[indices].shape[0] > 0:
            past_boxes.append(positive[indices])
            
        if len(past_boxes) > 10:
            past_boxes.pop(0)
        gt = cv2.imread(gt_path + filename_map[filename])
        # gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        frame = cv2.hconcat([frame, img])
        frame_bottom = cv2.hconcat([gt, np.array(diff)])
        frame = cv2.vconcat([frame, frame_bottom])
        
        if len(valid_boxes) > 0:
            gt_binary = (gt.mean(-1) > 0).astype(np.uint8)
            confusion.update(torch.tensor(gt_binary), torch.tensor(mask))
            print(confusion.get_f1(), confusion.get_iou(), confusion.get_precision(), confusion.get_recall())
            wandb.log({"f1": confusion.get_f1(),
                       "iou": confusion.get_iou(),
                        "precision": confusion.get_precision(),
                        "recall": confusion.get_recall(),
                        "frame": wandb.Image(frame)
                        })
            
        # pylab.clf()
        # pylab.imshow(frame)
        # pylab.show(block=False)
        # pylab.pause(0.0001)
        video_writer.write(frame)
        index += 1
    
video_writer.release()
os.system("ffmpeg -i out.mp4 -c:v libx264 out_x264.mp4 -y > /dev/null 2>&1")