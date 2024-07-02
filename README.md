<div align ="center">
<img src="assets/logo.jpg" width="20%">
<h1> EVF-SAM </h1>
<h3> Early Vision-Language Fusion for Text-Prompted Segment Anything Model </h3>

[Yuxuan Zhang](https://github.com/CoderZhangYx)<sup>1,\*</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ&hl=zh-CN)<sup>1,\*</sup>, Lei Liu<sup>2</sup>, Heng Liu<sup>2</sup>, Longjin Ran<sup>2</sup>,
<br>
Xiaoxin Chen<sup>2</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1,📧</sup>

<sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> vivo AI Lab

(\* equal contribution, 📧 corresponding author)

[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2406.20076)

</div>


## Highlight
<div align ="center">
<img src="assets/architecture.jpg">
</div>

* EVF-SAM extends SAM's capabilities with text-prompted segmentation, achieving high accuracy in Referring Expression Segmentation.  
* EVF-SAM is designed for efficient computation, enabling rapid inference in few seconds per image on a T4 GPU.


## Updates
- [ ] Release code
- [ ] Release weights
- [ ] Release demo


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


## Weights
<table class="center">
<tr>
  <td style="text-align:center;"><b>Model</b></td>
  <td style="text-align:center;"><b>SAM</b></td>
  <td style="text-align:center;"><b>Multimodal Encoder</b></td>
  <td style="text-align:center;"><b>Params</b></td>
  <td style="text-align:center;"><b>RefCOCO(cIoU)</b></td>
</tr>

<tr>
  <td style="text-align:center;"><b>EVF-SAM</b></td>
  <td style="text-align:center;"><b>SAM-H</b></td>
  <td style="text-align:center;"><b>BEIT-3-L</b></td>
  <td style="text-align:center;"><b>1.32B</b></td>
  <td style="text-align:center;"><b>83.7</b></td>
</tr>

<!-- <tr>
  <td style="text-align:center;"><b> evf-sam-fix </b></td>
  <td style="text-align:center;"><b>SAM-H</b></td>
  <td style="text-align:center;"><b>BEIT-3-L</b></td>
  <td style="text-align:center;"><b>1.32B</b></td>
  <td style="text-align:center;"><b>82.9</b></td>
</tr> -->

<tr>
  <td style="text-align:center;"><b>EVF-EfficientSAM</b></td>
  <td style="text-align:center;"><b>EfficientSAM-S</b></td>
  <td style="text-align:center;"><b>BEIT-3-L</b></td>
  <td style="text-align:center;"><b>700M</b></td>
  <td style="text-align:center;"><b>83.5</b></td>
</tr>

<tr>
  <td style="text-align:center;"><b>EVF-EfficientSAM</b></td>
  <td style="text-align:center;"><b>EfficientSAM-T</b></td>
  <td style="text-align:center;"><b>BEIT-3-B</b></td>
  <td style="text-align:center;"><b>232M</b></td>
  <td style="text-align:center;"><b>80.0</b></td>
</tr>
</table>


## Citation

```bibtex
@article{EVFSAM,
    title={EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model},
    author={Zhang, Yuxuan and Cheng, Tianheng and Hu, Rui and Liu, Lei and Liu, Heng and Ran, Longjin and Chen, Xiaoxin and Liu, Wenyu and Wang, Xinggang},
    journal={arXiv:2406.20076},
    year={2024}
}
```
