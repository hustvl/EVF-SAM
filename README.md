<div align ="center">
<img src="assets/logo.jpg" width="20%">
<h1> 📷 EVF-SAM </h1>
<h3> Early Vision-Language Fusion for Text-Prompted Segment Anything Model </h3>

[Yuxuan Zhang](https://github.com/CoderZhangYx)<sup>1,\*</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ&hl=zh-CN)<sup>1,\*</sup>, Lei Liu<sup>2</sup>, Heng Liu<sup>2</sup>, Longjin Ran<sup>2</sup>, Xiaoxin Chen<sup>2</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1,📧</sup>

<sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> vivo AI Lab

(\* equal contribution, 📧 corresponding author)

[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2406.20076)
[![🤗 HuggingFace models](https://img.shields.io/badge/HuggingFace🤗-Models-orange)](https://huggingface.co/YxZhang/)  
[![🤗 HuggingFace Demo](https://img.shields.io/badge/EVF_SAM-🤗_HF_Demo-orange)](https://huggingface.co/spaces/wondervictor/evf-sam)
[![🤗 HuggingFace Demo](https://img.shields.io/badge/EVF_SAM_2-🤗_HF_Demo-orange)](https://huggingface.co/spaces/wondervictor/evf-sam2)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hustvl/EVF-SAM/blob/main/inference_image.ipynb)

</div>

## News
We have expanded our EVF-SAM to powerful [SAM-2](https://github.com/facebookresearch/segment-anything-2). Besides improvements on image prediction, our new model also performs well on video prediction (powered by SAM-2). Only at the expense of a simple image training process on RES datasets, we find our EVF-SAM has zero-shot video text-prompted capability. Try our code!

## Highlight
<div align ="center">
<img src="assets/architecture.jpg">
</div>

* EVF-SAM extends SAM's capabilities with text-prompted segmentation, achieving high accuracy in Referring Expression Segmentation.  
* EVF-SAM is designed for efficient computation, enabling rapid inference in few seconds per image on a T4 GPU.


## Updates
- [x] Release code
- [x] Release weights
- [x] Release demo 👉 [🤗 evf-sam](https://huggingface.co/spaces/wondervictor/evf-sam)
- [x] Release code and weights based on SAM-2
- [x] Update demo supporting SAM-2👉 [🤗 evf-sam2](https://huggingface.co/spaces/wondervictor/evf-sam2)
- [x] release new checkpoint supporting body part segmentation and semantic level segmentation.
- [ ] update demo supporting multitask


## Visualization 
<table class="center">
<tr>
  <td style="text-align:center;"><b>Input text</b></td>
  <td style="text-align:center;"><b>Input image</b></td>
  <td style="text-align:center;"><b>Output</b></td>
</tr>
<tr>
  <td width=20% style="text-align:center;"><b>"zebra top left"</b></td>
  <td><img src="assets/zebra.jpg"></td>
  <td><img src="assets/zebra_vis.png"></td>
</tr> 

<tr>
  <td width=20% style="text-align:center;"><b>"a pizza with a yellow sign on top of it"</b></td>
  <td><img src="assets/pizza.jpg"></td>
  <td><img src="assets/pizza_vis.png"></td>
</tr> 

<tr>
  <td width=20% style="text-align:center;"><b>"the broccoli closest to the ketchup bottle"</b></td>
  <td><img src="assets/food.jpg"></td>
  <td><img src="assets/food_vis.png"></td>
</tr> 

<tr>
  <td width=20% style="text-align:center;"><b>"bus going to south common"</b></td>
  <td><img src="assets/bus.jpg"></td>
  <td><img src="assets/bus_vis.png"></td>
</tr> 

<tr>
  <td width=20% style="text-align:center;"><b>"3carrots in center with ice and greenn leaves"</b></td>
  <td><img src="assets/carrots.jpg"></td>
  <td><img src="assets/carrots_vis.png"></td>
</tr> 

</table>


## Installation
1. Clone this repository  
2. Install [pytorch](https://pytorch.org/) for your cuda version. **Note** that torch>=2.0.0 is needed if you are to use SAM-2, and torch>=2.2 is needed if you want to enable flash-attention. (We use torch==2.0.1 with CUDA 11.7 and it works fine.)
3. pip install -r requirements.txt
4. If you are to use the video prediction function, run:
```
cd model/segment_anything_2
python setup.py build_ext --inplace
```


## Weights
<table class="center">
<tr>
  <td style="text-align:center;"><b>Name</b></td>
  <td style="text-align:center;"><b>SAM</b></td>
  <td style="text-align:center;"><b>BEIT-3</b></td>
  <td style="text-align:center;"><b>Params</b></td>
  <td style="text-align:center;"><b>Prompt Encoder & Mask Decoder
  <td style="text-align:center;"><b>Reference Score</b></td>
</tr>
    
<tr>
  <td style="text-align:center;"><a href="https://huggingface.co/YxZhang/evf-sam-multitask">EVF-SAM-multitask</a></td>
  <td style="text-align:center;"><b>SAM-H</b></td>
  <td style="text-align:center;"><b>BEIT-3-L</b></td>
  <td style="text-align:center;"><b>1.32B</b></td>
  <td style="text-align:center;"><b>train</b></td>
  <td style="text-align:center;"><b>84.2</b></td>
</tr>
    
<tr>
  <td style="text-align:center;"><a href="https://huggingface.co/YxZhang/evf-sam2-multitask">EVF-SAM2-multitask</a></td>
  <td style="text-align:center;"><b>SAM-2-L</b></td>
  <td style="text-align:center;"><b>BEIT-3-L</b></td>
  <td style="text-align:center;"><b>898M</b></td>
  <td style="text-align:center;"><b>freeze</b></td>
  <td style="text-align:center;"><b>83.2</b></td>
</tr>

<tr>
  <td style="text-align:center;"><a href="https://huggingface.co/YxZhang/evf-sam">EVF-SAM</a></td>
  <td style="text-align:center;"><b>SAM-H</b></td>
  <td style="text-align:center;"><b>BEIT-3-L</b></td>
  <td style="text-align:center;"><b>1.32B</b></td>
  <td style="text-align:center;"><b>train</b></td>
  <td style="text-align:center;"><b>83.7</b></td>
</tr>

<tr>
  <td style="text-align:center;"><a href="https://huggingface.co/YxZhang/evf-sam2">EVF-SAM2</a></td>
  <td style="text-align:center;"><b>SAM-2-L</b></td>
  <td style="text-align:center;"><b>BEIT-3-L</b></td>
  <td style="text-align:center;"><b>898M</b></td>
  <td style="text-align:center;"><b>freeze</b></td>
  <td style="text-align:center;"><b>83.6</b></td>
</tr>



<tr>
  <td style="text-align:center;"><b>EVF-Effi-SAM-L </b></td>
  <td style="text-align:center;"><b>EfficientSAM-S</b></td>
  <td style="text-align:center;"><b>BEIT-3-L</b></td>
  <td style="text-align:center;"><b>700M</b></td>
  <td style="text-align:center;"><b>train</b></td>
  <td style="text-align:center;"><b>83.5</b></td>
</tr>

<tr>
  <td style="text-align:center;"><b>EVF-Effi-SAM-B </b></td>
  <td style="text-align:center;"><b>EfficientSAM-T</b></td>
  <td style="text-align:center;"><b>BEIT-3-B</b></td>
  <td style="text-align:center;"><b>232M</b></td>
  <td style="text-align:center;"><b>train</b></td>
  <td style="text-align:center;"><b>80.0</b></td>
</tr>
</table>

1. -multimask checkpoints are only available with commits>=9d00853, while other checkpoints are available with commits<9d00853

2. -multimask checkpoints are jointly trained on Ref, ADE20k, Object365, PartImageNet, humanparsing, pascal part datasets. These checkpoints is able to segment part (e.g., hair, arm), background object (e.g., sky, ground), and semantic-level masks.

## Inference
### 1. image prediction
```
python inference.py  \
  --version <path to evf-sam> \
  --precision='fp16' \
  --vis_save_path "<path to your output direction>" \
  --model_type <"ori" or "effi" or "sam2", depending on your loaded ckpt>   \
  --image_path <path to your input image> \
  --prompt <customized text prompt>
```
`--load_in_8bit` and `--load_in_4bit` is **optional**  
for example: 
```
python inference.py  \
  --version YxZhang/evf-sam2 \
  --precision='fp16' \
  --vis_save_path "vis" \
  --model_type sam2   \
  --image_path "assets/zebra.jpg" \
  --prompt "zebra top left"
```

### 2. video prediction  
firstly slice video into frames
```
ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <frame_dir>/'%05d.jpg'
```
then:
```
python inference_video.py  \
  --version <path to evf-sam2> \
  --precision='fp16' \
  --vis_save_path "vis/" \
  --image_path <frame_dir>   \
  --prompt <customized text prompt>   \
  --model_type sam2
```
you can use frame2video.py to concat the predicted frames to a video.

## Demo
image demo
```
python demo.py <path to evf-sam>
```
video demo
```
python demo_video.py <path to evf-sam2>
```

## Data preparation
Referring segmentation datasets: [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip), [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip), [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip), [refCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip) ([saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip)) and [COCO2014train](http://images.cocodataset.org/zips/train2014.zip)  
```
├── dataset
│   ├── refer_seg
│   │   ├── images
│   │   |   ├── saiapr_tc-12 
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
```

## Evaluation
```
torchrun --standalone --nproc_per_node <num_gpus> eval.py   \
    --version <path to evf-sam> \
    --dataset_dir <path to your data root>   \
    --val_dataset "refcoco|unc|val" \
    --model_type <"ori" or "effi" or "sam2", depending on your loaded ckpt>
```

## Acknowledgement
We borrow some codes from [LISA](https://github.com/dvlab-research/LISA/tree/main), [unilm](https://github.com/microsoft/unilm), [SAM](https://github.com/facebookresearch/segment-anything), [EfficientSAM](https://github.com/yformer/EfficientSAM), [SAM-2](https://github.com/facebookresearch/segment-anything-2).

## Citation
```bibtex
@article{zhang2024evfsamearlyvisionlanguagefusion,
      title={EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model}, 
      author={Yuxuan Zhang and Tianheng Cheng and Rui Hu and Lei Liu and Heng Liu and Longjin Ran and Xiaoxin Chen and Wenyu Liu and Xinggang Wang},
      year={2024},
      eprint={2406.20076},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.20076}, 
}
```
