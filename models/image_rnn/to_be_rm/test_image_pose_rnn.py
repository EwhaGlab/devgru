import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet

import os
from os.path import dirname, abspath
BASE_DIR = '/home/hankm/python_ws/viznav/depth-nav'
import sys
sys.path.append(BASE_DIR)

from models.base_model import BaseModel
from models.image_rnn.gru_model import GRUModel 

        
        
obs_encoding_size = 512
odom_encoding_size = 64
goal_encoding_size = obs_encoding_size
learn_angle = True
num_ch = 1
context_size = 5
len_traj_pred = 5
#self.late_fusion = late_fusion

obs_encoder = EfficientNet.from_name("efficientnet-b0" , in_channels=num_ch) # context
num_obs_features = obs_encoder._fc.in_features    # 1280
goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=num_ch*2) # goal and curr obs

num_goal_features = goal_encoder._fc.in_features  # 1280


if num_obs_features != obs_encoding_size:             # 1280 != 512  (obs_encoding_size = 512)
    compress_obs_enc = nn.Linear(num_obs_features, obs_encoding_size) # linear layer (1280 --> 512)
else:
    compress_obs_enc = nn.Identity()

if num_goal_features != goal_encoding_size:           # 1280 != 512  (goal_encoding_size == obs_encoding_size)
    compress_goal_enc = nn.Linear(num_goal_features, goal_encoding_size) # linear layer (1280 --> 512)
else:
    compress_goal_enc = nn.Identity()

if learn_angle is True:
    num_action_params = 4
else:
    num_action_params = 2

odom_encoder = nn.Sequential(nn.Linear(num_action_params, 128), nn.ReLU(), nn.Linear(128, odom_encoding_size))
goal_pose_encoder = nn.Sequential(nn.Linear(num_action_params, 128), nn.ReLU(), nn.Linear(128, odom_encoding_size))
obs_fusion_layer = nn.Sequential(nn.Linear(obs_encoding_size + odom_encoding_size, obs_encoding_size), nn.ReLU(), nn.Dropout(0.1) )
goal_fusion_layer = nn.Sequential(nn.Linear(goal_encoding_size + odom_encoding_size, goal_encoding_size), nn.ReLU(), nn.Dropout(0.1) )

# gru takes [batch, seq, feature]
#self.depth_nav = nn.GRU(input_size = self.obs_encoding_size, hidden_size = self.obs_encoding_size, batch_first=True)
gru_rnn = GRUModel(  input_size  = obs_encoding_size,  hidden_size = obs_encoding_size,  seq_len = context_size + 2)

dist_pred_len = int( num_action_params / 2 )
dist_predictor = nn.Sequential( nn.Linear(32, dist_pred_len) )

# len_traj_pred = 5 .. defined in base_model 
action_predictor = nn.Sequential(nn.Linear(32, len_traj_pred * num_action_params))



########################################################################################################3
#                       forward
########################################################################################################

    expected_ch = num_ch * (context_size + 1)
    assert obs_images.shape[1] == expected_ch, f"Expected {expected_ch} channels for observation images but got {obs_images.shape[1]}"

    # goal_depth: [B, C, H, W] where C is 1
    tuple_obs_images = torch.split(obs_images, num_ch, dim=1) # ( [B, ch, H, W], ..., [B, ch, H, W] )
    goal_concat = torch.concat( (tuple_obs_images[-1], goal_image), dim=1)
    goal_encoding = goal_encoder.extract_features(goal_concat)
    # goal_encoding : [B, 1280, 2, 2]
    goal_encoding = goal_encoder._avg_pooling(goal_encoding)
    # goal_encoding : [B, 1280, 1, 1]
    if goal_encoder._global_params.include_top:    # is True
        goal_encoding = goal_encoding.flatten(start_dim=1)
        #goal_encoding : [B, 1280*1*1]
        goal_encoding = goal_encoder._dropout(goal_encoding)
        #goal_encoding : [B, 1280*1*1]
    # currently, the size of goal_encoding is [batch_size, num_goal_features]... [B, 1280]
    goal_encoding = compress_goal_enc(goal_encoding)
    
    if len(goal_encoding.shape) == 2:   # is True  b/c  [B, 512]
        goal_encoding = goal_encoding.unsqueeze(1)  #  insert a dim after B ---> [B, 1, 512]
    # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
    assert goal_encoding.shape[2] == goal_encoding_size    # 512 == 512
    
    tuple_obs_images = torch.split(obs_images, num_ch, dim=1) #

    # image size is [batch_size*self.context_size, ch, H, W]
    obs_images = torch.concat(tuple_obs_images, dim=0) # [B*Contxt, 1, H, W]

    # obs_image.shape: [Q, ch, H, W], where Q = B * context_size
    # EffNet---------> avg_pooling -------------> flatten --------> dropout --------> compression ->
    #       [Q,1280,2,2]           [Q,1280,1,1]           [Q,1280]          [Q,1280]               [Q,512]
    #       ..-------> reshape to (Q, 1, 512) -----> transpose to (1, Q, 512)
    # get the observation encoding
    obs_encoding = obs_encoder.extract_features(obs_images)
    # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
    obs_encoding = obs_encoder._avg_pooling(obs_encoding)
    # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
    if obs_encoder._global_params.include_top: # is True
        obs_encoding = obs_encoding.flatten(start_dim=1)        # [B*(contextsize+1), 1280, 1, 1] --> [B*(contextsize+1), 1280]
        obs_encoding = obs_encoder._dropout(obs_encoding)  # [B*(contextsize+1), 1280]

    obs_encoding = compress_obs_enc(obs_encoding) # [B*(contextsize+1), 1280] --> [B*(contextsize+1), 512]
    # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
    # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
    obs_encoding = obs_encoding.reshape((context_size+1, -1, obs_encoding_size)) # [context+1, B, 512]
    obs_encoding = torch.transpose(obs_encoding, 0, 1)  # [B, context+1, 512]

    # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

    # obs fusion
    obs_odom_encoding = odom_encoder(context_actions)  # (B, num_context_actions, odom_encoding_size)
    fused_obs = torch.cat([obs_encoding, obs_odom_encoding], dim=-1)
    obs_fused_token = obs_fusion_layer(fused_obs)

    # goal fusion
    goal_pose_encoding = goal_pose_encoder(goal_pos)
    fused_goal = torch.cat([goal_encoding, goal_pose_encoding], dim=-1)
    goal_fused_token = goal_fusion_layer(fused_goal)

    # concatenate the goal encoding to the observation encoding
    tokens = torch.cat((obs_fused_token, goal_fused_token), dim=1)  # [B, 7, 512]
    
    # gru model
    final_repr = gru_rnn( tokens )       # [B, 32]

    action_pred = action_predictor(final_repr) # [B, num_params * len_traj_pred ] , ex) [B, 8] if we pred 4 wpts
    action_pred = action_pred.reshape( (action_pred.shape[0], len_trajectory_pred, num_action_params) ) # [B, len_traj_pred, num_params]
    action_pred = action_pred.squeeze()

    if (learn_angle):
        action_pred[..., 2:] = torch.nn.functional.normalize(action_pred[..., 2:].clone(), dim=2)  # [B, len_traj, num_params]

#    return  action_pred  #dist_pred, action_pred














