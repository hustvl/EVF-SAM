import os
import random
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

from model.segment_anything.utils.transforms import ResizeLongestSide

from torchvision import transforms
from utils.dataset import Resize

'''
categories
{'id': 0, 'name': 'Quadruped Head', 'supercategory': 'Quadruped'}, 
1: {'id': 1, 'name': 'Quadruped Body', 'supercategory': 'Quadruped'}, 
2: {'id': 2, 'name': 'Quadruped Foot', 'supercategory': 'Quadruped'}, 
3: {'id': 3, 'name': 'Quadruped Tail', 'supercategory': 'Quadruped'}, 
4: {'id': 4, 'name': 'Biped Head', 'supercategory': 'Biped'}, 
5: {'id': 5, 'name': 'Biped Body', 'supercategory': 'Biped'}, 
6: {'id': 6, 'name': 'Biped Hand', 'supercategory': 'Biped'}, 
7: {'id': 7, 'name': 'Biped Foot', 'supercategory': 'Biped'}, 
8: {'id': 8, 'name': 'Biped Tail', 'supercategory': 'Biped'}, 
9: {'id': 9, 'name': 'Fish Head', 'supercategory': 'Fish'}, 
10: {'id': 10, 'name': 'Fish Body', 'supercategory': 'Fish'}, 
11: {'id': 11, 'name': 'Fish Fin', 'supercategory': 'Fish'}, 
12: {'id': 12, 'name': 'Fish Tail', 'supercategory': 'Fish'}, 
13: {'id': 13, 'name': 'Bird Head', 'supercategory': 'Bird'}, 
14: {'id': 14, 'name': 'Bird Body', 'supercategory': 'Bird'}, 
15: {'id': 15, 'name': 'Bird Wing', 'supercategory': 'Bird'}, 
16: {'id': 16, 'name': 'Bird Foot', 'supercategory': 'Bird'}, 
17: {'id': 17, 'name': 'Bird Tail', 'supercategory': 'Bird'}, 
18: {'id': 18, 'name': 'Snake Head', 'supercategory': 'Snake'}, 
19: {'id': 19, 'name': 'Snake Body', 'supercategory': 'Snake'}, 
20: {'id': 20, 'name': 'Reptile Head', 'supercategory': 'Reptile'}, 
21: {'id': 21, 'name': 'Reptile Body', 'supercategory': 'Reptile'}, 
22: {'id': 22, 'name': 'Reptile Foot', 'supercategory': 'Reptile'}, 
23: {'id': 23, 'name': 'Reptile Tail', 'supercategory': 'Reptile'}, 
24: {'id': 24, 'name': 'Car Body', 'supercategory': 'Car'}, 
25: {'id': 25, 'name': 'Car Tier', 'supercategory': 'Car'}, 
26: {'id': 26, 'name': 'Car Side Mirror', 'supercategory': 'Car'}, 
27: {'id': 27, 'name': 'Bicycle Body', 'supercategory': 'Bicycle'}, 
28: {'id': 28, 'name': 'Bicycle Head', 'supercategory': 'Bicycle'}, 
29: {'id': 29, 'name': 'Bicycle Seat', 'supercategory': 'Bicycle'}, 
30: {'id': 30, 'name': 'Bicycle Tier', 'supercategory': 'Bicycle'}, 
31: {'id': 31, 'name': 'Boat Body', 'supercategory': 'Boat'}, 
32: {'id': 32, 'name': 'Boat Sail', 'supercategory': 'Boat'}, 
33: {'id': 33, 'name': 'Aeroplane Head', 'supercategory': 'Aeroplane'}, 
34: {'id': 34, 'name': 'Aeroplane Body', 'supercategory': 'Aeroplane'}, 
35: {'id': 35, 'name': 'Aeroplane Engine', 'supercategory': 'Aeroplane'}, 
36: {'id': 36, 'name': 'Aeroplane Wing', 'supercategory': 'Aeroplane'}, 
37: {'id': 37, 'name': 'Aeroplane Tail', 'supercategory': 'Aeroplane'}, 
38: {'id': 38, 'name': 'Bottle Mouth', 'supercategory': 'Bottle'}, 
39: {'id': 39, 'name': 'Bottle Body', 'supercategory': 'Bottle'}}
'''

class PartImageNetDataset(torch.utils.data.Dataset):
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
        
        self.DATA_DIR = os.path.join(base_image_dir, "PartImageNet", "images", "train")
        self.ANN_DIR = os.path.join(base_image_dir, "PartImageNet", "annotations", "train")
        self.meta = COCO(os.path.join(self.ANN_DIR, "train.json"))
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
        image_path = os.path.join(self.DATA_DIR, image_info["file_name"])

        ann_ids_per_img = self.meta.getAnnIds(image_id)
        anns = self.meta.loadAnns(ann_ids_per_img)

        record = {}
        for ann in anns:
            caption = self.meta.cats[ann["category_id"]]["name"]
            try:
                mask = self.meta.annToMask(ann)
            except:
                # there are several unavailable anns in partimagenet dataset.
                continue
            # we convert body part data from instance-level to semantic-level
            if not caption in record:
                record[caption] = mask
            else:
                record[caption] += mask
        if not record:
            return self.__getitem__(0)
        masks = list(record.values())
        sents = list(record.keys())

        if len(masks) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(masks))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(masks)))

        masks = np.stack(masks, 0)
        masks = masks[sampled_inds]
        sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sents = ["[semantic] "+ _.lower() for _ in sents]

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
