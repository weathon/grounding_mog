import os
import cv2
from transformers import Owlv2Processor, Owlv2ForObjectDetection


# root_path = "sim/in/"
root_path = "MOV_1242/"

videos = list(set(["_".join(i.split("_")[:-1]) for i in os.listdir(root_path)]))
# print(videos)
frames = sorted([i for i in os.listdir(root_path) if i.startswith("frame")])
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
from sam2.build_sam import build_sam2_camera_predictor
from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = ".sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

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
video_writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (img0.shape[1], img0.shape[0]))
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
index = 0
for frame in tqdm(frames[0:3000:10]):
    img = cv2.imread(root_path + frame)
    bgsub.apply(img)
    bg = bgsub.getBackgroundImage().astype(float)
    diff = cv2.absdiff(img.astype(float), bg) * 3 #too large cannot see detial
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    
    diff = Image.fromarray(diff)
    inputs = processor(text=texts, images=diff, return_tensors="pt", padding="longest").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.Tensor([diff.size[::-1]])
    
    frame = np.array(diff)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)
    
    i = 0
    text = texts[i]
    boxes, logits, phrases = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    
    positive = boxes[phrases == 0]
    positive_logits = logits[phrases == 0]
    indices = torchvision.ops.nms(positive, positive_logits, 0.3)
    
    # for box, logit in zip(positive[indices], positive_logits[indices]):
    #     x1, y1, x2, y2 = box.int().tolist()
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,255), 2)
    #     # check all boxes with sam mask, if high iou, do not resegment it (use the tracking mask), otherwise send to SAM


    valid_boxes = []    
    for current_box in boxes[indices]:
        matched_frames = 0
        
        for past_frame_boxes in past_boxes[max(-10, -len(past_boxes)):]: 
            ious = torchvision.ops.box_iou(current_box.unsqueeze(0), past_frame_boxes)
            print(ious)
            if (ious > 0.3).any():
                matched_frames += 1
                
        if matched_frames >= 2:
            valid_boxes.append(current_box)
    valid_boxes = torch.stack(valid_boxes) if valid_boxes else torch.empty((0, 4), device=positive.device)
    for idx, box in enumerate(torch.nn.functional.relu((valid_boxes.int()))): # use relu to avoid negative values
        x1, y1, x2, y2 = box.tolist()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # if not if_init:
            predictor.load_first_frame(frame)
            if_init = True
            _, out_obj_ids, masks = predictor.add_new_prompt(bbox=[[x1, y1], [x2, y2]], frame_idx=0, obj_id=1)
            # always update using current box if any, if not still show it, so we need this outside the for as well

            # else:
            # out_obj_ids, masks = predictor.track(frame)

        # predictor.set_image(frame)
        # masks, _, _ = predictor.predict(box = box[None,:], multimask_output=False)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        # cv2.putText(frame, "valid box", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)
        print(masks.max(), masks.min())
        mask = masks[0][0]>0
        mask = mask.cpu().numpy()
        # # fill wholes by closing
        # kernel = np.ones((5,5),np.uint8)
        # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0

        frame[mask,0] = 255
            
    if positive[indices].shape[0] > 0:
        past_boxes.append(positive[indices])
        
    if len(past_boxes) > 10:
        past_boxes.pop(0)
    pylab.clf()
    pylab.imshow(frame)
    pylab.show(block=False)
    pylab.pause(0.0001)
    video_writer.write(frame)
    index += 1
    
video_writer.release()
os.system("ffmpeg -i out.mp4 -c:v libx264 out_x264.mp4 -y > /dev/null 2>&1")