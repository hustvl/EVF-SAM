import gradio as gr
from inference import sam_preprocess, beit3_preprocess
from transformers import AutoTokenizer
import torch
import numpy as np
import sys
import time

version = sys.argv[1]
if "effi" in version:
    model_type = "effi"
elif "sam2" in version:
    model_type = "sam2"
else:
    model_type = "ori"

tokenizer = AutoTokenizer.from_pretrained(
        version,
        padding_side="right",
        use_fast=False,
    )

kwargs = {
    "torch_dtype": torch.half,
}
if model_type=="ori":
    from model.evf_sam import EvfSamModel
    model = EvfSamModel.from_pretrained(version, low_cpu_mem_usage=True, **kwargs).cuda().eval()
elif model_type=="effi":
    from model.evf_effisam import EvfEffiSamModel
    model = EvfEffiSamModel.from_pretrained(version, low_cpu_mem_usage=True, **kwargs).cuda().eval()
elif model_type=="sam2":
    from model.evf_sam2 import EvfSam2Model
    model = EvfSam2Model.from_pretrained(version, low_cpu_mem_usage=True, **kwargs)
    del model.visual_model.memory_encoder
    del model.visual_model.memory_attention
    model = model.cuda().eval()


@torch.no_grad()
def pred(image_np, prompt, semantic_type):
    # end = time.time()
    original_size_list = [image_np.shape[:2]]

    image_beit = beit3_preprocess(image_np, 224).to(dtype=model.dtype, device=model.device)

    image_sam, resize_shape = sam_preprocess(image_np, model_type=model_type)
    image_sam = image_sam.to(dtype=model.dtype, device=model.device)

    if semantic_type:
        prompt = "[semantic] " + prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

    # infer
    pred_mask = model.inference(
        image_sam.unsqueeze(0),
        image_beit.unsqueeze(0),
        input_ids,
        resize_list=[resize_shape],
        original_size_list=original_size_list,
    )
    pred_mask = pred_mask.detach().cpu().numpy()[0]
    pred_mask = pred_mask > 0

    visualization = image_np.copy()
    visualization[pred_mask] = (
        image_np * 0.5
        + pred_mask[:, :, None].astype(np.uint8) * np.array([220, 120, 50]) * 0.5
    )[pred_mask]
    # print(time.time() - end)
    return visualization/255.0, pred_mask.astype(np.float16)

demo = gr.Interface(
    fn=pred,
    inputs=[
        gr.components.Image(type="numpy", label="Image", image_mode="RGB"), 
        gr.components.Textbox(label="Prompt", info="Use a phrase or sentence to describe the object you want to segment. Currently we only support English"),
        gr.components.Checkbox(False, label="semantic level", info="check this if you want to segment body parts or background or multi objects (only available with latest evf-sam checkpoint)")],        
    outputs=[
        gr.components.Image(type="numpy", label="visulization"), 
        gr.components.Image(type="numpy", label="mask")],
    examples=[["assets/zebra.jpg", "zebra top left"], ["assets/bus.jpg", "bus going to south common"], ["assets/carrots.jpg", "3carrots in center with ice and greenn leaves"]],
    title="EVF-SAM referring expression segmentation",
    allow_flagging="never"
)
# demo.launch()
demo.launch(
    share=False,
    server_name="0.0.0.0",
    server_port=10001
)
