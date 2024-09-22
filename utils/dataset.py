import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils

from model.segment_anything.utils.transforms import ResizeLongestSide
from .refer import REFER
from torchvision import transforms
import json
from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image

class Resize:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        return np.array(resize(to_pil_image(image), (self.target_length, self.target_length), antialias=None))


def collate_fn(
    batch, tokenizer=None, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_evf_list = []
    masks_list = []
    label_list = []
    resize_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_evf,
        masks,
        label,
        resize,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_evf_list.append(images_evf)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        sampled_classes_list.extend(sampled_classes)
        cnt += len(sampled_classes)
        offset_list.append(cnt)
        inferences.append(inference)

    try:
        input_ids = [
            tokenizer(prompt, return_tensors="pt").input_ids[0]
            for prompt in sampled_classes_list
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
    # TinyCLIP
    except TypeError:
        input_ids = [
            tokenizer(prompt) for prompt in sampled_classes_list
        ]
        input_ids = torch.cat(input_ids, dim=0)

    
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_evf": torch.stack(images_evf_list, dim=0),
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        dataset="obj365||refer_seg",
        sample_rate=[1,3],
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        model_type="ori",
        transform=ResizeLongestSide(1024),
    ):
        self.transform=transform
        self.model_type = model_type
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        kwargs = dict(
            base_image_dir = base_image_dir,
            precision = precision,
            image_size = image_size,
            num_classes_per_sample = num_classes_per_sample,
            model_type = self.model_type,
            transform = self.transform
        )
        for dataset in self.datasets:
            if dataset == "refer_seg":
                from .refer_seg_dataset import ReferSegDataset
                self.all_datasets.append(
                    ReferSegDataset(refer_seg_data=refer_seg_data, **kwargs)
                )
            elif dataset == "ade20k":
                from .ade20k_dataset import Ade20kDataset
                self.all_datasets.append(
                    Ade20kDataset(**kwargs)
                )
            elif dataset == "obj365":
                from .o365_dataset import Obj365Dataset
                self.all_datasets.append(
                    Obj365Dataset(**kwargs)
                )
            elif dataset == "PartImageNet":
                from .partimagenet_dataset import PartImageNetDataset
                self.all_datasets.append(
                    PartImageNetDataset(**kwargs)
                )
            elif dataset == "humanparsing":
                from .humanparsing_dataset import HumanParsingDataset
                self.all_datasets.append(
                    HumanParsingDataset(**kwargs)
                )
            elif dataset == "pascal_part":
                from .pascal_part_dataset import PascalPartDataset
                self.all_datasets.append(
                    PascalPartDataset(**kwargs)
                )
            else:
                raise NotImplementedError("unknown dataset {}".format(dataset))

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        val_dataset,
        image_size=224,
        model_type="ori",
        transform=ResizeLongestSide(1024)
    ):
        if model_type=="ori":
            assert isinstance(transform, ResizeLongestSide)
        else:
            assert isinstance(transform, Resize)

        self.model_type = model_type
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 3:
            ds, splitBy, split = splits
            base_image_dir = os.path.join(base_image_dir, "refer_seg")
            refer_api = REFER(base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        elif val_dataset=="ade":
            with open(os.path.join(base_image_dir, "ade20k", "ade20k_classes.json")) as f:
                self.categories = json.load(f)

            self.labels = sorted(
                os.listdir(
                    os.path.join(base_image_dir, "ade20k", "annotations", "validation")
                )
            )
            self.data_type="ade"

        self.transform = transform
        self.image_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), interpolation=3, antialias=None), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
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
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            masks = []
            for i, ann_id in enumerate(ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = maskUtils.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = maskUtils.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)

        elif self.data_type == "ade":
            label_path = random.choice(self.labels)
            image_path = os.path.join(self.base_image_dir, "ade20k", "images", "validation", label_path[:-4]+".jpg")
            label_path = os.path.join(self.base_image_dir, "ade20k", "annotations", "validation", label_path)

            semantic_map = cv2.imread(label_path, 0)

            semantic_map[semantic_map == 0] = 255
            semantic_map -= 1
            semantic_map[semantic_map == 254] = 255

            cats = np.unique(semantic_map).tolist()
            if 255 in cats:
                cats.remove(255)
            if len(cats) == 0:
                return self.__getitem__(0)

            sents = []
            masks = []
            for ann in cats:
                m = semantic_map==ann
                m = m.astype(np.uint8)
                masks.append(m)
                caption = self.categories[ann]
                sents.append(caption)
            sents = ["[semantic] " + _ for _ in sents]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess image for evf
        image_evf = self.image_preprocessor(image)

        # preprocess image for sam
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if not isinstance(masks, torch.Tensor):
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_evf,
            masks,
            labels,
            resize,
            sents,
            inference,
        )
