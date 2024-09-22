import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.segment_anything.utils.transforms import ResizeLongestSide
from torchvision import transforms
from utils.dataset import Resize


class Ade20kDataset(torch.utils.data.Dataset):
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

        self.DATA_ROOT = os.path.join(base_image_dir, "ade20k")
        with open(os.path.join(self.DATA_ROOT, "ade20k_classes.json")) as f:
            self.categories = json.load(f)

        self.labels = sorted(
            os.listdir(os.path.join(self.DATA_ROOT, "annotations", "training"))
        )

    def __len__(self):
        return len(self.labels)

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
        label_path = random.choice(self.labels)
        image_path = os.path.join(self.DATA_ROOT, "images", "training", label_path[:-4]+".jpg")
        label_path = os.path.join(self.DATA_ROOT, "annotations", "training", label_path)
        semantic_map = cv2.imread(label_path, 0)

        semantic_map[semantic_map == 0] = 255
        semantic_map -= 1
        semantic_map[semantic_map == 254] = 255

        cats = np.unique(semantic_map).tolist()
        if 255 in cats:
            cats.remove(255)
        if len(cats) == 0:
            return self.__getitem__(0)

        if len(cats) >= self.num_classes_per_sample:
            sampled_anns = np.random.choice(
                cats, size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_anns = cats

        sents = []
        masks = []
        for ann in sampled_anns:
            mask = semantic_map==ann
            mask = mask.astype(np.uint8)
            masks.append(mask)
            caption = self.categories[ann]
            sents.append(caption)
        masks = np.stack(masks, 0)
        sents = ["[semantic] " + _ for _ in sents]

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
