import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.vint_rgbd.self_attention import MultiLayerDecoder

class ViNT_RGBD(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(ViNT_RGBD, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        #self.depth_obs_encoding_size = obs_encoding_size
        #self.rgb_goal_encoding_size = obs_encoding_size
        #self.depth_goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion  # false by default
        if obs_encoder.split("-")[0] == "efficientnet":
            self.rgb_obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.depth_obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=1) # context
            self.num_rgb_obs_features = self.rgb_obs_encoder._fc.in_features    # 1280
            self.num_depth_obs_features = self.depth_obs_encoder._fc.in_features  # 1280
            if self.late_fusion:    # false by default
                self.rgb_goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
                self.depth_goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=1)
            else:
                self.rgb_goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal (rgb)
                self.depth_goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=2) # obs+goal (depth)
            self.num_rgb_goal_features = self.rgb_goal_encoder._fc.in_features          # 1280
            self.num_depth_goal_features = self.depth_goal_encoder._fc.in_features  # 1280
        else:
            raise NotImplementedError
        
        if self.num_rgb_obs_features != self.obs_encoding_size:            # obs_encoding_size = 512
            self.compress_rgb_obs_enc = nn.Linear(self.num_rgb_obs_features, self.obs_encoding_size)
        else:
            self.compress_rgb_obs_enc = nn.Identity()

        if self.num_depth_obs_features != self.obs_encoding_size :         # obs_encoding_size = 512
            self.compress_depth_obs_enc = nn.Linear(self.num_depth_obs_features, self.obs_encoding_size )
        else:
            self.compress_depth_obs_enc = nn.Identity()
# goal
        if self.num_rgb_goal_features != self.goal_encoding_size:
            self.compress_rgb_goal_enc = nn.Linear(self.num_rgb_goal_features, self.goal_encoding_size)
        else:
            self.compress_rgb_goal_enc = nn.Identity()

        if self.num_depth_goal_features != self.goal_encoding_size :
            self.compress_depth_goal_enc = nn.Linear(self.num_depth_goal_features, self.goal_encoding_size )
        else:
            self.compress_depth_goal_enc = nn.Identity()

        # if self.num_depth_goal_features != self.goal_encoding_size/2:
        #     self.compress_depth_goal_enc = nn.Linear(self.num_depth_goal_features, self.goal_encoding_size)
        # else:
        #     self.compress_rgb_goal_enc = nn.Identity()

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len= (self.context_size+2)*2, #self.context_size+2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,)

        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),)

        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),   # =len_traj_pred .. defined in base_model
        )

    def obsgoal_fused_rgb_encode(self, obs_rgb: torch.tensor, goal_rgb: torch.tensor):
        if self.late_fusion:
            rgb_goal_encoding = self.rgb_goal_encoder.extract_features(goal_rgb)
        else:
            obsgoal_rgb = torch.cat([obs_rgb[:, 3 * self.context_size:, :, :], goal_rgb],
                                    dim=1)  # ch 15:18 (curr) + goal  #[B, 6, H, W]
            rgb_goal_encoding = self.rgb_goal_encoder.extract_features(obsgoal_rgb)  # [B, 1280, 2, 2]
        rgb_goal_encoding = self.rgb_goal_encoder._avg_pooling(rgb_goal_encoding)  # [B, 1280, 1, 1]

        if self.rgb_goal_encoder._global_params.include_top:  # True
            rgb_goal_encoding = rgb_goal_encoding.flatten(start_dim=1)  # [B, 1280]
            rgb_goal_encoding = self.rgb_goal_encoder._dropout(rgb_goal_encoding)
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        rgb_goal_encoding = self.compress_rgb_goal_enc(rgb_goal_encoding)  # [B, 1280] --> [B, 512]
        if len(rgb_goal_encoding.shape) == 2:  # True
            rgb_goal_encoding = rgb_goal_encoding.unsqueeze(1)  # [B, 1, 1280]
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert rgb_goal_encoding.shape[2] == self.goal_encoding_size  # 512 == 512  is True

        return rgb_goal_encoding

    def obs_rgb_econde(self, obs_rgb: torch.tensor):
        # obs_img.shape: [Q, 3, H, W], where Q = B * context_size
        # EffNet---------> avg_pooling -------------> flatten --------> dropout --------> compression ->
        #       [Q,1280,2,2]           [Q,1280,1,1]           [Q,1280]          [Q,1280]               [Q,512]
        #       ..-------> reshape to (Q, 1, 512) -----> transpose to (1, Q, 512)
        # get the observation encoding

        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_rgb = torch.split(obs_rgb, 3, dim=1)  # tuple of obs rgb imgs ... len(obs_rgb) == 6

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_rgb = torch.concat(obs_rgb, dim=0) # [B*cs, 3, H, W]

        rgb_obs_encoding = self.rgb_obs_encoder.extract_features(obs_rgb)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        rgb_obs_encoding = self.rgb_obs_encoder._avg_pooling(rgb_obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.rgb_obs_encoder._global_params.include_top:
            rgb_obs_encoding = rgb_obs_encoding.flatten(start_dim=1)
            rgb_obs_encoding = self.rgb_obs_encoder._dropout(rgb_obs_encoding)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        rgb_obs_encoding = self.compress_rgb_obs_enc(rgb_obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        rgb_obs_encoding = rgb_obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        rgb_obs_encoding = torch.transpose(rgb_obs_encoding, 0, 1)

        return rgb_obs_encoding

    def obsgoal_fused_depth_econde(self, obs_depth: torch.tensor, goal_depth: torch.tensor):
        if self.late_fusion:
            depth_goal_encoding = self.depth_goal_encoder.extract_features(goal_depth)
        else:
            obsgoal_depth = torch.cat([obs_depth[:, self.context_size:, :, :], goal_depth], dim=1)  # ch 15:18 (curr) + goal
            depth_goal_encoding = self.depth_goal_encoder.extract_features(obsgoal_depth)
        depth_goal_encoding = self.depth_goal_encoder._avg_pooling(depth_goal_encoding)

        if self.depth_goal_encoder._global_params.include_top:
            depth_goal_encoding = depth_goal_encoding.flatten(start_dim=1)
            depth_goal_encoding = self.depth_goal_encoder._dropout(depth_goal_encoding)
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        depth_goal_encoding = self.compress_depth_goal_enc(depth_goal_encoding)
        if len(depth_goal_encoding.shape) == 2:
            depth_goal_encoding = depth_goal_encoding.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert depth_goal_encoding.shape[2] == self.goal_encoding_size

        return depth_goal_encoding

    def obs_depth_encode(self, obs_depth: torch.tensor):
        # split the observation into context based on the context size
        # image size is [batch_size, 1*self.context_size, H, W]
        obs_depth = torch.split(obs_depth, 1, dim=1)
        # image size is [batch_size*self.context_size, 1, H, W]
        obs_depth = torch.concat(obs_depth, dim=0)
        depth_obs_encoding = self.depth_obs_encoder.extract_features(obs_depth)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]  i.e., [B*Cont, 1280, 2, 2]
        depth_obs_encoding = self.depth_obs_encoder._avg_pooling(depth_obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.depth_obs_encoder._global_params.include_top:
            depth_obs_encoding = depth_obs_encoding.flatten(start_dim=1)
            depth_obs_encoding = self.depth_obs_encoder._dropout(depth_obs_encoding)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        depth_obs_encoding = self.compress_depth_obs_enc(depth_obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        depth_obs_encoding = depth_obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        depth_obs_encoding = torch.transpose(depth_obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]
        return depth_obs_encoding

    def forward(
        self, obs_rgb: torch.tensor, obs_depth: torch.tensor, goal_rgb: torch.tensor, goal_depth: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # encode obsgoal
        # obsgoal_img.shape: [B, 6, H, W]
        # EffNet---------> avg_pooling -------------> flatten ---------> dropout --------> compression ->
        #    [B,1280,2,2]               [B,1280,1,1]           [B,1280]          [B,1280]               [B,512]
        # get the fused observation and goal encoding

        rgb_goal_encoding = self.obsgoal_fused_rgb_encode(obs_rgb, goal_rgb)            # [B, 6, 512]
        rgb_obs_encoding = self.obs_rgb_econde(obs_rgb)                                 # [B, 1, 512]
        depth_goal_encoding = self.obsgoal_fused_depth_econde(obs_depth, goal_depth)    # [B, 1, 512]
        depth_obs_encoding = self.obs_depth_encode(obs_depth)                           # [B, 6, 512]

        # concatenate the goal encoding to the observation encoding
        rgbd_tokens = torch.cat((rgb_obs_encoding, depth_goal_encoding, rgb_goal_encoding, depth_obs_encoding), dim=1)
        # Currently [B, 14, 512]
        rgbd_final_repr = self.decoder(rgbd_tokens) # transformer decoder based on self attentions
        # currently, the size is [batch_size, 32]

################

        dist_pred = self.dist_predictor(rgbd_final_repr)
        action_pred = self.action_predictor(rgbd_final_repr)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred