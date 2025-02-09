

import os
import cv2
from transformers import AutoProcessor, AutoModelForCausalLM

videos = ["_".join(i.split("_")[:-1]) for i in os.listdir("sim/in")]
# print(videos)
frames = sorted([i for i in os.listdir("sim/in") if i.startswith("vid17_")])
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
import matplotlib.animation as animation


# use sam to do tracking
# use reverse? cannot include all list
# pre encode text
texts = ["semi-transparent steam cloud plume on black background"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)

img0 = cv2.imread("sim/in/" + frames[0])
history_state = []
bgsub = cv2.createBackgroundSubtractorMOG2(history=30)
video_writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (img0.shape[1], img0.shape[0]))



# def image2box(image):
#     prompt = ["<REGION_PROPOSAL>"]

#     inputs = processor(text=prompt, images=[image] * len(prompt), return_tensors="pt", padding=True).to(device, torch_dtype)
#     generated_ids = model.generate(
#         input_ids=inputs["input_ids"],
#         pixel_values=inputs["pixel_values"],
#         max_new_tokens=1024,
#         num_beams=3,
#     )
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)

#     res = processor.post_process_generation(generated_text[0], task="<REGION_PROPOSAL>", image_size=(image.width, image.height))
#     res = res["<REGION_PROPOSAL>"]
#     # print(res)
#     return res["bboxes"]

# use point prompt?


for frame in tqdm(frames):
    img = cv2.imread("sim/in/" + frame)
    bgsub.apply(img)
    bg = bgsub.getBackgroundImage().astype(float)
    diff = cv2.absdiff(img.astype(float), bg) * 10 #too large cannot see detial
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    
    diff = Image.fromarray(diff)
    
    boxes = image2box(diff)
    diff = np.array(diff)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(diff, (x1, y1), (x2, y2), (255,0,255), 2)
        pylab.imshow(diff)
        pylab.show(block=False)
    video_writer.write(diff)
    
video_writer.release()
os.system("ffmpeg -i out.mp4 -c:v libx264 out_x264.mp4 -y > /dev/null 2>&1")




# import torch
# import cv2

# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to("cuda")
# prompt = cv2.resize(cv2.imread("prompt.png"), (320//14 * 14, 320//14 * 14)).astype(float)/256
# test = cv2.resize(cv2.imread("test.png"), (320//14 * 14, 320//14 * 14)).astype(float)/256
# prompt = torch.tensor(prompt).permute(2, 0, 1).unsqueeze(0).to("cuda").to(torch.float32)
# test = torch.tensor(test).permute(2, 0, 1).unsqueeze(0).to("cuda").to(torch.float32)
# # print(dinov2_vits14(prompt).shape)
# # print(dinov2_vits14.get_intermediate_layers(prompt)[0].reshape(1, 320//14, 320//14, -1).shape)

# prompt_feature = torch.nn.functional.normalize(dinov2_vits14(prompt), dim=-1)
# test_feature = torch.nn.functional.normalize(dinov2_vits14.get_intermediate_layers(test)[0].reshape(1, 320//14, 320//14, -1), dim=-1)


# sim = torch.einsum("bf,bwhf->bwh", prompt_feature, test_feature).cpu().detach().numpy()[0]
# print(sim.shape)
# import pylab
# pylab.imshow(sim)
# pylab.show()