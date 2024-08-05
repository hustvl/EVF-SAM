import sys
import time
import os
import gradio as gr
import torch
import numpy as np
import cv2
from tqdm import tqdm
from transformers import AutoTokenizer
from inference import beit3_preprocess

version = sys.argv[1]
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
tokenizer = AutoTokenizer.from_pretrained(
        version,
        padding_side="right",
        use_fast=False,
    )

from model.evf_sam2_video import EvfSam2Model
kwargs = {
    "torch_dtype": torch.half,
}
model = EvfSam2Model.from_pretrained(version, low_cpu_mem_usage=True, **kwargs).cuda().eval()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.float16)
def pred(video_path, prompt):
    # end = time.time()
    os.system("rm -rf demo_temp")
    os.makedirs("demo_temp/input_frames", exist_ok=True)
    os.system("ffmpeg -i {} -q:v 2 -start_number 0 demo_temp/input_frames/'%05d.jpg'".format(video_path))
    input_frames = sorted(os.listdir("demo_temp/input_frames"))
    image_np = cv2.imread("demo_temp/input_frames/00000.jpg")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    height, width, channels = image_np.shape

    image_beit = beit3_preprocess(image_np, 224).to(dtype=model.dtype, device=model.device)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

    # infer
    output = model.inference(
        "demo_temp/input_frames",
        image_beit.unsqueeze(0),
        input_ids,
    )
    # save visualization
    video_writer = cv2.VideoWriter("demo_temp/out.mp4", fourcc, 30, (width,height))
    pbar = tqdm(input_frames)
    pbar.set_description("generating video: ")
    for i, file in enumerate(pbar):
        img = cv2.imread(os.path.join("demo_temp/input_frames", file))
        vis = img + np.array([0, 0, 128]) * output[i][1].transpose(1,2,0)
        vis = np.clip(vis, 0, 255)
        vis = np.uint8(vis)
        video_writer.write(vis)
    video_writer.release()
    return "demo_temp/out.mp4"
    
    # print(time.time() - end)

demo = gr.Interface(
    fn=pred,
    inputs=[
        gr.components.Video(label="Input video"), 
        gr.components.Textbox(label="Prompt", info="Use a phrase or sentence to describe the object you want to segment. Currently we only support English")],
    outputs=[
        gr.components.Video(label="Output video")],
    title="EVF-SAM2 referring expression segmentation",
    description="Please don't upload long video. It takes about 1 min to process 150 frames.",
    allow_flagging="never"
)
demo.launch()
# demo.launch(
#     share=False,
#     server_name="0.0.0.0",
#     server_port=10001
# )