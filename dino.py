

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

img0 = cv2.imread("sim/in/" + frames[0])
history_state = []
bgsub = cv2.createBackgroundSubtractorMOG2(history=30)
video_writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (img0.shape[1], img0.shape[0]))
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to("cuda")

class Processor:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to("cuda")
        self.std = torch.tensor([0.229, 0.224, 0.225]).to("cuda")
        
    def __call__(self, img):
        img = cv2.resize(img, (320//14 * 14, 320//14 * 14)).astype(float)/256
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to("cuda").to(torch.float32)
        img = (img - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return img        

processor = Processor()
query_image = cv2.imread("prompt.png")
query_image = processor(query_image)
query_feature = torch.nn.functional.normalize(model(query_image), dim=-1)



rpn_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
rpn_odel = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)

def image2box(image):
    prompt = ["<REGION_PROPOSAL>"]

    inputs = rpn_processor(text=prompt, images=[image] * len(prompt), return_tensors="pt", padding=True).to(device, torch_dtype)
    generated_ids = rpn_odel.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        temperature=3.0,
        do_sample=True,
    )
    generated_text = rpn_processor.batch_decode(generated_ids, skip_special_tokens=False)

    res = rpn_processor.post_process_generation(generated_text[0], task="<REGION_PROPOSAL>", image_size=(image.width, image.height))
    res = res["<REGION_PROPOSAL>"]
    # print(res)
    return res["bboxes"]


for frame in tqdm(frames[200:]):
    img = cv2.imread("sim/in/" + frame)
    bgsub.apply(img)
    bg = bgsub.getBackgroundImage().astype(float)
    diff = cv2.absdiff(img.astype(float), bg) * 10 #too large cannot see detial
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    
    diff = Image.fromarray(diff)
    
    diff = np.array(diff)
    diff_pt = processor(diff)
    diff_feature = torch.nn.functional.normalize(model.get_intermediate_layers(diff_pt)[0].reshape(1, 320//14, 320//14, -1), dim=-1)
    boxes = image2box(Image.fromarray(diff))
    diff = np.array(diff)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(diff, (x1, y1), (x2, y2), (255,0,255), 2)
    pylab.clf()
    pylab.imshow(diff)
    pylab.show(block=False)
    pylab.pause(0.00001)
    
    video_writer.write(diff)
    
video_writer.release()
os.system("ffmpeg -i out.mp4 -c:v libx264 out_x264.mp4 -y > /dev/null 2>&1")


