from typing import List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from .segment_anything_2.sam2.build_sam import build_sam2, build_sam2_video_predictor
from .unilm.beit3.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from .configuration_evf import EvfConfig
from .segment_anything_2.sam2.utils.misc import load_video_frames
from collections import OrderedDict



def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

class EvfSam2Model(PreTrainedModel):
    config_class = EvfConfig
    def __init__(
        self,
        config,
        **kwargs
    ):
        super(EvfSam2Model, self).__init__(config)

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.encoder_pretrained = kwargs.get("encoder_pretrained", None)
        self.dice_loss_weight = kwargs.get("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.get("bce_loss_weight", None)
        self.train_mask_decoder = kwargs.get("train_mask_decoder", False)
        self.train_prompt_encoder = kwargs.get("train_prompt_encoder", False)
        self.initialize_evf_modules(config)
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def initialize_evf_modules(self, config):
        # SAM
        if config.sam_scale=="large":
            self.visual_model = build_sam2_video_predictor("sam2_hiera_l.yaml", self.vision_pretrained, device=None)
        elif config.sam_scale=="tiny":
            self.visual_model = build_sam2_video_predictor("sam2_hiera_t.yaml", self.vision_pretrained, device=None)
        else:
            raise NotImplementedError
        
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if self.train_mask_decoder:
            self.visual_model.sam_mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        if self.train_prompt_encoder:
            self.visual_model.sam_prompt_encoder.no_mask_embed.requires_grad_(True)
            
        # beit-3
        if self.config.mm_extractor_scale == "base":
            beit_config = _get_base_config()
        elif self.config.mm_extractor_scale == "large":
            beit_config = _get_large_config()
        else:
            raise AttributeError(f"model config should contain key 'mm_extractor_scale', with value 'base' or 'large'.")

        self.mm_extractor = BEiT3Wrapper(beit_config)
        if self.encoder_pretrained is not None:
            beit_state_dict = torch.load(self.encoder_pretrained)["model"]
            self.mm_extractor.load_state_dict(
                beit_state_dict, 
                strict=False
            )

        for param in self.mm_extractor.parameters():
            param.requires_grad = True
                
        # Projection layer
        in_dim = config.hidden_size
        assert in_dim==beit_config.encoder_embed_dim, \
            f"projection layer dim {in_dim} mismatch with mm_extractor dim {beit_config.encoder_embed_dim}"
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks


    def inference(
            self,
            video_path,
            images_evf,
            input_ids,
        ):
        predictor = self.visual_model
        inference_state = predictor.init_state(video_path=video_path)
        predictor.reset_state(inference_state)

        output = self.mm_extractor.beit3(visual_tokens=images_evf, textual_tokens=input_ids, text_padding_position=torch.zeros_like(input_ids))

        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        _, out_obj_ids, out_mask_logits = predictor.add_new_text(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            text=feat
        )

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        return video_segments
  

AutoConfig.register("evf", EvfConfig)
AutoModelForCausalLM.register(EvfConfig, EvfSam2Model)