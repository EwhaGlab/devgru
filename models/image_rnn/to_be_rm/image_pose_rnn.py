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
from models.image_rnn.gru_model import GRUModel
#from vint_train.models.vint.self_attention import MultiLayerDecoder

class ImagePoseRNN(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = False,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        odom_encoding_size: Optional[int] = 64,
        num_ch: Optional[int] = 1,  # input images channel size
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
            obs_encoding_size (int): size of the encoding of the observation depths
            goal_encoding_size (int): size of the encoding of the goal depths
        """
        super(ImagePoseRNN, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.learn_angle = learn_angle
        self.num_ch = num_ch
        #self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=self.num_ch) # context
            self.num_obs_features = self.obs_encoder._fc.in_features    # 1280
            #if self.late_fusion:
                #self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            #else:
                #self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=self.num_ch*2) # goal and curr obs
            self.num_goal_features = self.goal_encoder._fc.in_features  # 1280
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.obs_encoding_size:             # 1280 != 512  (obs_encoding_size = 512)
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size) # linear layer (1280 --> 512)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:           # 1280 != 512  (goal_encoding_size == obs_encoding_size)
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size) # linear layer (1280 --> 512)
        else:
            self.compress_goal_enc = nn.Identity()

        if self.learn_angle is True:
            self.num_action_params = 4
        else:
            self.num_action_params = 2

        self.odom_encoding_size = odom_encoding_size
        self.odom_encoder = nn.Sequential(nn.Linear(self.num_action_params, 128), nn.ReLU(), nn.Linear(128, self.odom_encoding_size))
        self.goal_pose_encoder = nn.Sequential(nn.Linear(self.num_action_params, 128), nn.ReLU(), nn.Linear(128, self.odom_encoding_size))
        self.obs_fusion_layer = nn.Sequential(nn.Linear(self.obs_encoding_size + self.odom_encoding_size, self.obs_encoding_size), nn.ReLU(), nn.Dropout(0.1) )
        self.goal_fusion_layer = nn.Sequential(nn.Linear(self.goal_encoding_size + self.odom_encoding_size, self.goal_encoding_size), nn.ReLU(), nn.Dropout(0.1) )

        # gru takes [batch, seq, feature]
        #self.depth_nav = nn.GRU(input_size = self.obs_encoding_size, hidden_size = self.obs_encoding_size, batch_first=True)
        self.image_pose_rnn = GRUModel( input_size = self.obs_encoding_size,
                                        hidden_size= self.obs_encoding_size,
                                        seq_len    = self.context_size + 2)

        dist_pred_len = int( self.num_action_params ) #/ 2 )  [dx, dy, qw, qz]
        self.dist_predictor = nn.Sequential( nn.Linear(32, dist_pred_len) )

        # len_traj_pred = 5 .. defined in base_model 
        self.action_predictor = nn.Sequential(
             nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

        self.collision_predictor = nn.Linear(32, 1)

    def forward(
        self, obs_images: torch.tensor, goal_image: torch.tensor, context_actions: torch.tensor, goal_pos: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # obsgoal_img.shape: [B, 6, H, W]
        # EffNet---------> avg_pooling -------------> flatten ---------> dropout --------> compression ->
        #    [B,1280,2,2]               [B,1280,1,1]           [B,1280]          [B,1280]               [B,512]
        # get the fused observation and goal encoding
        #if self.late_fusion:
            #goal_encoding = self.goal_encoder.extract_features(goal_img)
        #else:
            #obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)   # ch 15:18 (curr) + goal
            #goal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        
        expected_ch = self.num_ch * (self.context_size + 1)
        assert obs_images.shape[1] == expected_ch, \
            f"Expected {expected_ch} channels for observation images but got {obs_images.shape[1]}"

        # goal_depth: [B, C, H, W] where C is 1
        tuple_obs_images = torch.split(obs_images, self.num_ch, dim=1) # ( [B, ch, H, W], ..., [B, ch, H, W] )
        goal_concat = torch.concat( (tuple_obs_images[-1], goal_image), dim=1)
        goal_encoding = self.goal_encoder.extract_features(goal_concat)
        # goal_encoding : [B, 1280, 2, 2]
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        # goal_encoding : [B, 1280, 1, 1]
        if self.goal_encoder._global_params.include_top:    # is True
            goal_encoding = goal_encoding.flatten(start_dim=1)
            #goal_encoding : [B, 1280*1*1]
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
            #goal_encoding : [B, 1280*1*1]
        # currently, the size of goal_encoding is [batch_size, num_goal_features]... [B, 1280]
        goal_encoding = self.compress_goal_enc(goal_encoding)
        # goal_encoding : [B, 512]
        
        if len(goal_encoding.shape) == 2:   # is True  b/c  [B, 512]
            goal_encoding = goal_encoding.unsqueeze(1)  #  insert a dim after B ---> [B, 1, 512]
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding.shape[2] == self.goal_encoding_size    # 512 == 512
        
        # split the observation into context based on the context size
        # image size is [batch_size, ch*self.context_size, H, W]
        tuple_obs_images = torch.split(obs_images, self.num_ch, dim=1) #

        # image size is [batch_size*self.context_size, ch, H, W]
        obs_images = torch.concat(tuple_obs_images, dim=0) # [B*Contxt, 1, H, W]

        # obs_image.shape: [Q, ch, H, W], where Q = B * context_size
        # EffNet---------> avg_pooling -------------> flatten --------> dropout --------> compression ->
        #       [Q,1280,2,2]           [Q,1280,1,1]           [Q,1280]          [Q,1280]               [Q,512]
        #       ..-------> reshape to (Q, 1, 512) -----> transpose to (1, Q, 512)
        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_images)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top: # is True
            obs_encoding = obs_encoding.flatten(start_dim=1)        # [B*(contextsize+1), 1280, 1, 1] --> [B*(contextsize+1), 1280]
            obs_encoding = self.obs_encoder._dropout(obs_encoding)  # [B*(contextsize+1), 1280]

        obs_encoding = self.compress_obs_enc(obs_encoding) # [B*(contextsize+1), 1280] --> [B*(contextsize+1), 512]
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size)) # [context+1, B, 512]
        obs_encoding = torch.transpose(obs_encoding, 0, 1)  # [B, context+1, 512]

        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # obs fusion
        obs_odom_encoding = self.odom_encoder(context_actions)
        fused_obs = torch.cat([obs_encoding, obs_odom_encoding], dim=-1)
        obs_fused_token = self.obs_fusion_layer(fused_obs)  #  obs_fused_token  shape is (B, context, 512)

        if len(goal_pos.shape) == 2:   # is True  b/c  [B, 64]
            goal_pos = goal_pos.unsqueeze(1)  #  insert a dim after B ---> [B, 1, 64]
        # goal fusion
        goal_pose_encoding = self.goal_pose_encoder(goal_pos)
        fused_goal = torch.cat([goal_encoding, goal_pose_encoding], dim=-1)
        goal_fused_token = self.goal_fusion_layer(fused_goal)

        # concatenate the goal encoding to the observation encoding
        tokens = torch.cat((obs_fused_token, goal_fused_token), dim=1)  # [B, 7, 512]
        
        # gru model
        final_repr = self.image_pose_rnn( tokens )       # [B, 32]
        
        # final_repr = self.decoder(tokens) # transformer decoder based on self attentions
        # currently, the size is [batch_size, 32]

        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr) # [B, num_params * len_traj_pred ] , ex) [B, 8] if we pred 4 wpts
        collision_pred = self.collision_predictor(final_repr).squeeze(-1)
        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape( (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params) ) # [B, len_traj_pred, num_params]
        action_pred = action_pred #.squeeze()
        #action_pred[..., :2] = torch.cumsum(action_pred[..., :2], dim=1)  # convert position deltas into waypoints
        # normalize output quaternion
        if (self.learn_angle):
            action_pred[..., 2:] = torch.nn.functional.normalize(action_pred[..., 2:].clone(), dim=2)  # [B, len_traj, num_params]

        #if self.learn_angle:
            #action_pred[:, :, 2:] = F.normalize(
                #action_pred[:, :, 2:].clone(), dim=-1
            #)  # normalize the angle prediction
        return  action_pred,  dist_pred, collision_pred  #, action_pred
