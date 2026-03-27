import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet

import os
from os.path import dirname, abspath
BASE_DIR = os.path.join(dirname(dirname(os.path.abspath(__file__))))
import sys
sys.path.append(BASE_DIR)

from models.base_model import BaseModel
#from models.image_rnn.gru_model import GRUModel
#from vint_train.models.vint.self_attention import MultiLayerDecoder

class DepthSGCollision(BaseModel):
    def __init__(
        self,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        num_ch: Optional[int] = 1,  # input images channel size
        pretrained: bool = False, #True,
        dropout_p: float = 0.2,
        freeze_backbone_bn: bool = False,
        fuse_hidden_size: Optional[int] = None,
    ) -> None:
        """
        Inputs:
          curr_depth: (B, C, H, W) depth or inverse-depth normalized to [0,1]
          sg_depth:   (B, C, H, W) depth or inverse-depth normalized to [0,1]

        Output:
          logit: (B,) collision logits (use sigmoid for probability)
        """
        super(DepthSGCollision, self).__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.freeze_backbone_bn = bool(freeze_backbone_bn)
        #self.goal_encoding_size = obs_encoding_size
        self.num_ch = num_ch

        if fuse_hidden_size is None:
            fuse_hidden_size = self.obs_encoding_size

        if obs_encoder.split("-")[0] == "efficientnet":

            if pretrained:
                self.obs_encoder = EfficientNet.from_pretrained(obs_encoder, in_channels=self.num_ch)
            else:
                self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=self.num_ch)

            self.num_obs_features = self.obs_encoder._fc.in_features  # 1280
            self.obs_encoder._dropout = nn.Identity()
            self.obs_encoder._fc = nn.Identity()
        else:
            raise NotImplementedError


        if self.num_obs_features != self.obs_encoding_size:             # 1280 != 512  (obs_encoding_size = 512)
            self.proj = nn.Sequential(
                nn.Linear(self.num_obs_features, obs_encoding_size),
                nn.SiLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ) # linear layer (1280 --> 512)

        else:
            self.proj = nn.Identity()

        fuse_in = obs_encoding_size * 2
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, obs_encoding_size),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        self.coll_head = nn.Linear(fuse_hidden_size, 1)  # output logit

        if self.freeze_backbone_bn:
            self._set_backbone_bn_eval()

    def _set_backbone_bn_eval(self):
        for m in self.obs_encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def forward(self, curr_depth: torch.Tensor, sg_depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            curr_depth: (B, C, H, W) normalized to [0,1]
            sg_depth:   (B, C, H, W) normalized to [0,1]
        Returns:
            logit: (B,) collision logits
        """
        if self.freeze_backbone_bn:
            self._set_backbone_bn_eval()

        f_curr = self.obs_encoder(curr_depth)           # (B, num_obs_features)
        f_sg = self.obs_encoder(sg_depth)

        z_curr = self.proj(f_curr)
        z_sg   = self.proj(f_sg) # (B, obs_encoding_size)

        # fuse
        #fused = torch.cat( [z_curr, z_sg, (z_curr - z_sg).abs(), z_curr * z_sg ], dim = -1)
        fused = torch.cat([z_curr, z_sg], dim=-1)
        h = self.fuse(fused)
        logit = self.coll_head(h).squeeze(-1)  # (B,)

        return logit

    @torch.no_grad()
    def infer_prob(self, depth: torch.Tensor) -> torch.Tensor:
        """Convenience: returns collision probability in [0,1]."""
        return torch.sigmoid(self.forward(depth))
