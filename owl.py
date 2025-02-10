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
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
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



boxes = [] 
# record boxes for past 10 frames. For each new frame, 
# if there is any boxes matched (>t_iou) for 8/10 past frames, 
# then it is a valid box. Send it to sam for tracking and segmentation
for frame in tqdm(frames[2000:5000]):
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
    
    for box, logit in zip(positive[indices], positive_logits[indices]):
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,255), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 1)


    # if len(positive) > 0:
    #     with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #         predictor.set_image(np.array(diff))
    #         masks, _, _ = predictor.predict(box=positive[None,:], multimask_output=False, return_logits=True)
    #         if len(masks.shape) > 3:
    #             masks = masks.sum(0)
    #         masks = masks.sum(0)
    #         if prev_masks is None:
    #             prev_masks = [masks]
    #         else:
    #             prev_masks.append(masks)
    #             if len(prev_masks) >5:
    #                 prev_masks.pop(0)
    #         print(len(prev_masks), prev_masks[0].max()) 
    #         # pylab.clf()
    #         # pylab.imshow(masks * 255)
    #         # pylab.show(block=False)
    #         # easy to rise but hard to fall
    # if prev_masks is not None:
    #     mask_ = np.median(sigmoid(prev_masks), axis=0) > 0.3
        # frame[mask_, 0] = 255 
    # even if false positive, if it then disappear, it is not a problem as sam will not be able to track it
    # if sam no mask for certain frame, then it is dropped 
    pylab.clf()
    pylab.imshow(frame)
    pylab.show(block=False)
    pylab.pause(0.0001)
    video_writer.write(frame)
    
video_writer.release()
os.system("ffmpeg -i out.mp4 -c:v libx264 out_x264.mp4 -y > /dev/null 2>&1")