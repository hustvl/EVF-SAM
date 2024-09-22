import os
import random
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import pandas as pd

from model.segment_anything.utils.transforms import ResizeLongestSide

from torchvision import transforms
from utils.dataset import Resize


class Obj365Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        model_type="ori",
        transform=ResizeLongestSide(1024),
    ):
        if model_type=="ori":
            assert isinstance(transform, ResizeLongestSide)
        else:
            assert isinstance(transform, Resize)
        self.model_type = model_type
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.precision = precision
        self.transform = transform
        self.image_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), interpolation=3, antialias=None), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        self.DATA_DIR = os.path.join(base_image_dir, "Object365", "raw", "Objects365_v1", "2019-08-02")
        self.meta = COCO(os.path.join(self.DATA_DIR, "o365_res_instances.json"))
        self.images = self.meta.getImgIds()
        self.total_imgs = len(self.images)
       

    def __len__(self):
        return self.total_imgs

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""  
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        if self.model_type=="ori":
            # Pad
            h, w = x.shape[-2:]
            padh = self.img_size - h
            padw = self.img_size - w
            x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image_id = random.choice(self.images)
        image_info = self.meta.loadImgs(image_id)[0]
        image_path = os.path.join(self.DATA_DIR, "train", image_info["file_name"])

        ann_ids_per_img = self.meta.getAnnIds(image_id)
        anns = self.meta.loadAnns(ann_ids_per_img)
        if len(anns) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(anns))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(anns)))
        sampled_anns = np.vectorize(anns.__getitem__)(sampled_inds).tolist()

        sents = []
        masks = []
        for ann in sampled_anns:
            seg = ann["segmentation"][0]
            caption = self.meta.cats[ann["category_id"]]["name"]
            mask = maskUtils.decode(seg)
            sents.append(caption)
            masks.append(mask)
        masks = np.stack(masks, 0)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess image for evf
        image_evf = self.image_preprocessor(image)

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_evf,
            masks,
            label,
            resize,
            sents,
        )
