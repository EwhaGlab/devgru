import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from depth_nav_train.models.base_model import BaseModel
from depth_nav_train.models.depth_rnn.gru_model import GRUModel
#from vint_train.models.vint.self_attention import MultiLayerDecoder

class DepthRNN(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = False,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        #late_fusion: Optional[bool] = False,
        #mha_num_attention_heads: Optional[int] = 2,
        #mha_num_attention_layers: Optional[int] = 2,
        #mha_ff_dim_factor: Optional[int] = 4,
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
        
        
        #super(DepthRNN, self).__init__(context_size, len_traj_pred, learn_angle)
        #self.obs_encoding_size = obs_encoding_size
        goal_encoding_size = obs_encoding_size
        #self.learn_angle = learn_angle

        #self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=1) # context
            num_obs_features = obs_encoder._fc.in_features    # 1280
            #if self.late_fusion:
                #self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            #else:
                #self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=1) # goal
            num_goal_features = goal_encoder._fc.in_features  # 1280
        else:
            raise NotImplementedError
        
        if num_obs_features != obs_encoding_size:               # 1280 != 512
            compress_obs_enc = nn.Linear(num_obs_features, obs_encoding_size) # linear layer that does 1280 --> 512
        else:
            compress_obs_enc = nn.Identity()
        
        if num_goal_features != goal_encoding_size:
            compress_goal_enc = nn.Linear(num_goal_features, goal_encoding_size)
        else:
            compress_goal_enc = nn.Identity()

        #self.decoder = MultiLayerDecoder(
            #embed_dim=self.obs_encoding_size,
            #seq_len=self.context_size+2,
            #output_layers=[256, 128, 64, 32],
            #nhead=mha_num_attention_heads,
            #num_layers=mha_num_attention_layers,
            #ff_dim_factor=mha_ff_dim_factor,
        #)
        
        # gru takes [batch, seq, feature] 
        #self.depth_nav = nn.GRU(input_size = self.obs_encoding_size, hidden_size = self.obs_encoding_size, batch_first=True)
        depth_rnn = GRUModel(   input_size  = obs_encoding_size,
                                hidden_size = obs_encoding_size,
                                seq_len     = context_size + 2)
        if learn_angle is True:
            num_action_params = 4
        else:
            num_action_params = 2

        dist_pred_len = int( num_action_params / 2 )
        dist_predictor = nn.Sequential( nn.Linear(32, dist_pred_len),)
        # len_traj_pred = 5 .. defined in base_model 
        action_predictor = nn.Sequential(
             nn.Linear(32, len_trajectory_pred * num_action_params),
        )

    def forward(
        self, obs_depths: torch.tensor, goal_depth: torch.tensor,
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
        
        # goal_depth: [B, 1, H, W]
        goal_encoding = goal_encoder.extract_features(goal_depth)
        # goal_encoding : [B, 1280, 2, 2]
        goal_encoding = goal_encoder._avg_pooling(goal_encoding)
        # goal_encoding : [B, 1280, 1, 1]
        if goal_encoder._global_params.include_top :    # is True
            goal_encoding = goal_encoding.flatten(start_dim=1)
            #goal_encoding : [B, 1280*1*1]
            goal_encoding = goal_encoder._dropout(goal_encoding)
            #goal_encoding : [B, 1280*1*1]
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        goal_encoding = compress_goal_enc(goal_encoding)
        # goal_encoding : [B, 512]
        
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding.shape[2] == goal_encoding_size
        
        # split the observation into context based on the context size
        # depth size is [batch_size, 1*self.context_size, H, W]
        obs_depths = torch.split(obs_depths, 1, dim=1)

        # depth size is [batch_size*self.context_size, 1, H, W]
        obs_depths = torch.concat(obs_depths, dim=0)

        # obs_depth.shape: [Q, 1, H, W], where Q = B * context_size
        # EffNet---------> avg_pooling -------------> flatten --------> dropout --------> compression ->
        #       [Q,1280,2,2]           [Q,1280,1,1]           [Q,1280]          [Q,1280]               [Q,512]
        #       ..-------> reshape to (Q, 1, 512) -----> transpose to (1, Q, 512)
        # get the observation encoding
        obs_encoding = obs_encoder.extract_features(obs_depths)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if obs_encoder._global_params.include_top: # is True
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = obs_encoder._dropout(obs_encoding)

        obs_encoding = compress_obs_enc(obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((context_size+1, -1, obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)  # [B, 7, 512]
        
        # gru model
        final_repr = depth_rnn( tokens )
        
        # final_repr = self.decoder(tokens) # transformer decoder based on self attentions
        # currently, the size is [batch_size, 32]

        dist_pred = dist_predictor(final_repr)
        action_pred = action_predictor(final_repr)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape( (action_pred.shape[0], len_trajectory_pred, num_action_params) )
        action_pred = action_pred.squeeze()
        #action_pred[..., :2] = torch.cumsum(action_pred[..., :2], dim=1)  # convert position deltas into waypoints
        # normalize output quaternion
        if (learn_angle):
            action_pred[..., 2:] = torch.nn.functional.normalize(action_pred[..., 2:].clone(), dim=1)

        #if self.learn_angle:
            #action_pred[:, :, 2:] = F.normalize(
                #action_pred[:, :, 2:].clone(), dim=-1
            #)  # normalize the angle prediction
        return dist_pred, action_pred
