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

# cats = {
#     {'id': 1, 'name': 'aeroplane:body'}, 2: {'id': 2, 'name': 'aeroplane:wing'}, 
#     3: {'id': 3, 'name': 'aeroplane:tail'}, 4: {'id': 4, 'name': 'aeroplane:wheel'}, 
#     5: {'id': 5, 'name': 'bicycle:wheel'}, 6: {'id': 6, 'name': 'bicycle:handlebar'}, 
#     7: {'id': 7, 'name': 'bicycle:saddle'}, 8: {'id': 8, 'name': 'bird:beak'}, 
#     9: {'id': 9, 'name': 'bird:head'}, 10: {'id': 10, 'name': 'bird:eye'}, 
#     11: {'id': 11, 'name': 'bird:leg'}, 12: {'id': 12, 'name': 'bird:foot'}, 
#     13: {'id': 13, 'name': 'bird:wing'}, 14: {'id': 14, 'name': 'bird:neck'}, 
#     15: {'id': 15, 'name': 'bird:tail'}, 16: {'id': 16, 'name': 'bird:torso'}, 
#     17: {'id': 17, 'name': 'bottle:body'}, 18: {'id': 18, 'name': 'bottle:cap'}, 
#     19: {'id': 19, 'name': 'bus:license plate', 'abbr': 'bus:liplate'}, 20: {'id': 20, 'name': 'bus:headlight'}, 
#     21: {'id': 21, 'name': 'bus:door'}, 22: {'id': 22, 'name': 'bus:mirror'}, 
#     23: {'id': 23, 'name': 'bus:window'}, 24: {'id': 24, 'name': 'bus:wheel'}, 
#     25: {'id': 25, 'name': 'car:license plate', 'abbr': 'car:liplate'}, 26: {'id': 26, 'name': 'car:headlight'}, 
#     27: {'id': 27, 'name': 'car:door'}, 28: {'id': 28, 'name': 'car:mirror'}, 
#     29: {'id': 29, 'name': 'car:window'}, 30: {'id': 30, 'name': 'car:wheel'}, 
#     31: {'id': 31, 'name': 'cat:head'}, 32: {'id': 32, 'name': 'cat:leg'}, 
#     33: {'id': 33, 'name': 'cat:ear'}, 34: {'id': 34, 'name': 'cat:eye'}, 
#     35: {'id': 35, 'name': 'cat:paw', 'abbr': 'cat:pa'}, 36: {'id': 36, 'name': 'cat:neck'}, 
#     37: {'id': 37, 'name': 'cat:nose'}, 38: {'id': 38, 'name': 'cat:tail'}, 
#     39: {'id': 39, 'name': 'cat:torso'}, 40: {'id': 40, 'name': 'cow:head'}, 
#     41: {'id': 41, 'name': 'cow:leg'}, 42: {'id': 42, 'name': 'cow:ear'}, 
#     43: {'id': 43, 'name': 'cow:eye'}, 44: {'id': 44, 'name': 'cow:neck'}, 
#     45: {'id': 45, 'name': 'cow:horn'}, 46: {'id': 46, 'name': 'cow:muzzle'}, 
#     47: {'id': 47, 'name': 'cow:tail'}, 48: {'id': 48, 'name': 'cow:torso'}, 
#     49: {'id': 49, 'name': 'dog:head'}, 50: {'id': 50, 'name': 'dog:leg'}, 
#     51: {'id': 51, 'name': 'dog:ear'}, 52: {'id': 52, 'name': 'dog:eye'}, 
#     53: {'id': 53, 'name': 'dog:paw', 'abbr': 'dog:pa'}, 54: {'id': 54, 'name': 'dog:neck'}, 
#     55: {'id': 55, 'name': 'dog:nose'}, 56: {'id': 56, 'name': 'dog:muzzle'}, 
#     57: {'id': 57, 'name': 'dog:tail'}, 58: {'id': 58, 'name': 'dog:torso'}, 
#     59: {'id': 59, 'name': 'horse:head'}, 60: {'id': 60, 'name': 'horse:leg'}, 
#     61: {'id': 61, 'name': 'horse:ear'}, 62: {'id': 62, 'name': 'horse:eye'}, 
#     63: {'id': 63, 'name': 'horse:neck'}, 64: {'id': 64, 'name': 'horse:muzzle'}, 
#     65: {'id': 65, 'name': 'horse:tail'}, 66: {'id': 66, 'name': 'horse:torso'}, 
#     67: {'id': 67, 'name': 'motorbike:wheel'}, 68: {'id': 68, 'name': 'motorbike:handlebar'}, 
#     69: {'id': 69, 'name': 'motorbike:headlight'}, 70: {'id': 70, 'name': 'motorbike:saddle'}, 
#     71: {'id': 71, 'name': 'person:hair'}, 72: {'id': 72, 'name': 'person:head'}, 
#     73: {'id': 73, 'name': 'person:ear'}, 74: {'id': 74, 'name': 'person:eye'}, 
#     75: {'id': 75, 'name': 'person:nose'}, 76: {'id': 76, 'name': 'person:neck'}, 
#     77: {'id': 77, 'name': 'person:mouth'}, 78: {'id': 78, 'name': 'person:arm'}, 
#     79: {'id': 79, 'name': 'person:hand'}, 80: {'id': 80, 'name': 'person:leg'}, 
#     81: {'id': 81, 'name': 'person:foot'}, 82: {'id': 82, 'name': 'person:torso'}, 
#     83: {'id': 83, 'name': 'potted plant:plant'}, 84: {'id': 84, 'name': 'potted plant:pot'}, 
#     85: {'id': 85, 'name': 'sheep:head'}, 86: {'id': 86, 'name': 'sheep:leg'}, 
#     87: {'id': 87, 'name': 'sheep:ear'}, 88: {'id': 88, 'name': 'sheep:eye'}, 
#     89: {'id': 89, 'name': 'sheep:neck'}, 90: {'id': 90, 'name': 'sheep:horn'}, 
#     91: {'id': 91, 'name': 'sheep:muzzle'}, 92: {'id': 92, 'name': 'sheep:tail'}, 
#     93: {'id': 93, 'name': 'sheep:torso'}
# }

class PascalPartDataset(torch.utils.data.Dataset):
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
        
        self.DATA_DIR = os.path.join(base_image_dir, "vlpart", "pascal_part", "VOCdevkit", "VOC2010", "JPEGImages")
        self.meta = COCO(os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json"))
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
        sents = ["[semantic] "+ _.replace(':', ' ') for _ in sents]

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
