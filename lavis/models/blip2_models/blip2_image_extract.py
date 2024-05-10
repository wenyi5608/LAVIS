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

@registry.register_model("blip2_image_extract")
class Blip2ImageFeatureExtract(Blip2Qformer):
    """
    BLIP2 Image Feature Extract model.
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_image_extract", "pretrained")
        >>> model = load_model("blip2_image_extract", "coco")
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

    def forward(self, image):
        assert (
            image is not None
        ), "Image is not provided for mode 'image' "
        # return query features
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(
            image_embeds.shape[0], -1, -1
        )
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
        )


