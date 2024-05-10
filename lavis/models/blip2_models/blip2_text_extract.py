"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

@registry.register_model("blip2_text_extract")
class Blip2TextFeatureExtract(Blip2Qformer):
    """
    BLIP2 Text Feature Extract model..
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_text_extract", "pretrained")
        >>> model = load_model("blip2_text_extract", "coco")
    """

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

    def forward(self, input_ids, attention_mask):
        assert (
            input_ids is not None
        ), " input_ids is None for mode 'text'"

        text_output = self.Qformer.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embeds = text_output.last_hidden_state
        text_features = self.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)
        return BlipOutputFeatures(
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
        )
