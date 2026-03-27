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

class DepthCollision(BaseModel):
    def __init__(
        self,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        num_ch: Optional[int] = 1,  # input images channel size
        pretrained: bool = False, #True,
        dropout_p: float = 0.2,
        freeze_backbone_bn: bool = False,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation depths
            goal_encoding_size (int): size of the encoding of the goal depths
        """
        super(DepthCollision, self).__init__()
        self.obs_encoding_size = obs_encoding_size
        self.freeze_backbone_bn = bool(freeze_backbone_bn)
        #self.goal_encoding_size = obs_encoding_size
        self.num_ch = num_ch

        if obs_encoder.split("-")[0] == "efficientnet":

            if pretrained:
                self.obs_encoder = EfficientNet.from_pretrained(obs_encoder, in_channels=self.num_ch)
            else:
                self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=self.num_ch)

            self.num_obs_features = self.obs_encoder._fc.in_features    # 1280
            self.obs_encoder._dropout = nn.Identity()
            self.obs_encoder._fc = nn.Identity()

#            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=self.num_ch*2) # goal and curr obs
#            self.num_goal_features = self.goal_encoder._fc.in_features  # 1280
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

        self.coll_head = nn.Linear(obs_encoding_size, 1)  # output logit

        if self.freeze_backbone_bn:
            self._set_backbone_bn_eval()

    def _set_backbone_bn_eval(self):
        for m in self.obs_encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, num_ch, H, W) depth or inverse-depth normalized to [0,1]
        Returns:
            logits: (B,) collision logits (apply sigmoid for probability)
        """
        if self.freeze_backbone_bn:
            self._set_backbone_bn_eval()

        feats = self.obs_encoder(depth)           # (B, num_obs_features)
        z = self.proj(feats)                      # (B, obs_encoding_size)
        logit = self.coll_head(z).squeeze(-1)  # (B,)
        return logit

    @torch.no_grad()
    def infer_prob(self, depth: torch.Tensor) -> torch.Tensor:
        """Convenience: returns collision probability in [0,1]."""
        return torch.sigmoid(self.forward(depth))
