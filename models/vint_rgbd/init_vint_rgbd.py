import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import MultiLayerDecoder


#class ViNT_RGBD(BaseModel):
    #def __init__(
        #self,
        #context_size: int = 5,
        #len_traj_pred: Optional[int] = 5,
        #learn_angle: Optional[bool] = True,
        #obs_encoder: Optional[str] = "efficientnet-b0",
        #obs_encoding_size: Optional[int] = 512,
        #late_fusion: Optional[bool] = False,
        #mha_num_attention_heads: Optional[int] = 2,
        #mha_num_attention_layers: Optional[int] = 2,
        #mha_ff_dim_factor: Optional[int] = 4,
    #) -> None:

#        super(ViNT_RGBD, self).__init__(context_size, len_traj_pred, learn_angle)
        
    context_size = 5
    len_traj_pred = 5
    learn_angle = True
    obs_encoder = "efficientnet-b0"
    obs_encoding_size = 512
    late_fusion = False
    mha_num_attention_heads = 2
    mha_num_attention_layers = 2
    mha_ff_dim_factor = 4
    
    len_traj_pred = 5     # defined in BaseModel
    if learn_angle:
        num_action_params = 4           # defined in BaseModel
    else:
        num_action_params = 2
        
        
    rgb_obs_encoding_size = obs_encoding_size
    depth_obs_encoding_size = obs_encoding_size
    rgb_goal_encoding_size = obs_encoding_size
    depth_goal_encoding_size = obs_encoding_size

    late_fusion = late_fusion  # false by default
    if obs_encoder.split("-")[0] == "efficientnet":
        rgb_obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
        depth_obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=1) # context
        num_rgb_obs_features = rgb_obs_encoder._fc.in_features        # 1280
        num_depth_obs_features = depth_obs_encoder._fc.in_features    # 1280
        if late_fusion:    # false by default
            rgb_goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            depth_goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=1)
        else:
            rgb_goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal (rgb)
            depth_goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=2) # obs+goal (depth)
        num_rgb_goal_features = rgb_goal_encoder._fc.in_features      # 1280
        num_depth_goal_features = depth_goal_encoder._fc.in_features  # 1280    / 2
    else:
        raise NotImplementedError
    
    if num_rgb_obs_features != obs_encoding_size:             # obs_encoding_size = 512
        compress_rgb_obs_enc = nn.Linear(num_rgb_obs_features, obs_encoding_size)
    else:
        compress_rgb_obs_enc = nn.Identity()

    if num_depth_obs_features != obs_encoding_size :         # obs_encoding_size = 512
        compress_depth_obs_enc = nn.Linear(num_depth_obs_features, obs_encoding_size )
    else:
        compress_depth_obs_enc = nn.Identity()
# goal
    if num_rgb_goal_features != obs_encoding_size :
        compress_rgb_goal_enc = nn.Linear(num_rgb_goal_features, obs_encoding_size)
    else:
        compress_rgb_goal_enc = nn.Identity()

    if num_depth_goal_features != obs_encoding_size :
        compress_depth_goal_enc = nn.Linear(num_depth_goal_features, obs_encoding_size )
    else:
        compress_depth_goal_enc = nn.Identity()

    # if self.num_depth_goal_features != self.goal_encoding_size/2:
    #     self.compress_depth_goal_enc = nn.Linear(self.num_depth_goal_features, self.goal_encoding_size)
    # else:
    #     self.compress_rgb_goal_enc = nn.Identity()

    decoder = MultiLayerDecoder(
        embed_dim= obs_encoding_size,
        seq_len= (context_size+2)*2,  #context_size+2,
        output_layers=[256, 128, 64, 32],
        nhead=mha_num_attention_heads,
        num_layers=mha_num_attention_layers,
        ff_dim_factor=mha_ff_dim_factor,)
    
    dist_predictor = nn.Sequential(nn.Linear(32, 1),)
    
    action_predictor = nn.Sequential(
        nn.Linear(32, len_traj_pred * num_action_params),   # =len_traj_pred .. defined in base_model
    )


########################################################################################

#def forward(
    #self, obs_rgb: torch.tensor, goal_rgb: torch.tensor, obs_depth: torch.tensor, goal_depth: torch.tensor
#) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

obs_rgb     = train_dataset[0][0] # 3 (rgb)  x 6 (context + curr) 
obs_depth   = train_dataset[0][1] # 1 (depth)x 6 (context + curr)
goal_rgb    = train_dataset[0][2] # 3 (rgb) x 1
goal_depth  = train_dataset[0][3] # 1 (depth) x 1

    def obsgoal_fused_rgb_encode(obs_rgb: torch.tensor, goal_rgb: torch.tensor):
        
        if( len(obs_rgb.shape) == 3):
            obs_rgb = obs_rgb[None,...]
        
        if( len(goal_rgb.shape) == 3):
            goal_rgb = goal_rgb[None,...]
        
        if late_fusion:
            rgb_goal_encoding = rgb_goal_encoder.extract_features(goal_rgb)
        else:
            obsgoal_rgb = torch.cat([obs_rgb[:, 3 * context_size:, :, :], goal_rgb], dim=1)  # ch 15:18 (curr) + goal  #[B, 6, H, W]
            rgb_goal_encoding = rgb_goal_encoder.extract_features(obsgoal_rgb)  # [B, 1280, 2, 2]
        rgb_goal_encoding = rgb_goal_encoder._avg_pooling(rgb_goal_encoding)  # [B, 1280, 1, 1]

        if rgb_goal_encoder._global_params.include_top:  # True
            rgb_goal_encoding = rgb_goal_encoding.flatten(start_dim=1)  # [B, 1280]
            rgb_goal_encoding = rgb_goal_encoder._dropout(rgb_goal_encoding)
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        rgb_goal_encoding = compress_rgb_goal_enc(rgb_goal_encoding)  # [B, 1280] --> [B, 512]
        if len(rgb_goal_encoding.shape) == 2:  # True
            rgb_goal_encoding = rgb_goal_encoding.unsqueeze(1)  # [B, 1, 1280]
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert rgb_goal_encoding.shape[2] == rgb_goal_encoding_size  # 512 == 512  is True
        return rgb_goal_encoding

    def obs_rgb_econde(obs_rgb: torch.tensor):
        # obs_img.shape: [Q, 3, H, W], where Q = B * context_size
        # EffNet---------> avg_pooling -------------> flatten --------> dropout --------> compression ->
        #       [Q,1280,2,2]           [Q,1280,1,1]           [Q,1280]          [Q,1280]               [Q,512]
        #       ..-------> reshape to (Q, 1, 512) -----> transpose to (1, Q, 512)
        # get the observation encoding

        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        if ( len(obs_rgb.shape) == 3 ):
            obs_rgb = obs_rgb[None, ...]
        
        obs_rgb = torch.split(obs_rgb, 3, dim=1)  # tuple of obs rgb imgs ... len(obs_rgb) == 6
        # image size is [batch_size*self.context_size, 3, H, W]
        obs_rgb = torch.concat(obs_rgb, dim=0)
        rgb_obs_encoding = rgb_obs_encoder.extract_features(obs_rgb)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        rgb_obs_encoding = rgb_obs_encoder._avg_pooling(rgb_obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if rgb_obs_encoder._global_params.include_top:
            rgb_obs_encoding = rgb_obs_encoding.flatten(start_dim=1)
            rgb_obs_encoding = rgb_obs_encoder._dropout(rgb_obs_encoding)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]
        rgb_obs_encoding = compress_rgb_obs_enc(rgb_obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        
        rgb_obs_encoding = rgb_obs_encoding.reshape((context_size+1, -1, obs_encoding_size))
        rgb_obs_encoding = torch.transpose(rgb_obs_encoding, 0, 1)
        return rgb_obs_encoding

    def obsgoal_fused_depth_econde(obs_depth: torch.tensor, goal_depth: torch.tensor):
        
        if( len(obs_depth.shape) == 3):
            obs_depth = obs_depth[None,...]
        
        if( len(goal_depth.shape) == 3):
            goal_depth = goal_depth[None,...]
        
        if late_fusion:
            depth_goal_encoding = depth_goal_encoder.extract_features(goal_depth)
        else:
            obsgoal_depth = torch.cat([obs_depth[:, context_size:, :, :], goal_depth], dim=1)  # ch 15:18 (curr) + goal
            depth_goal_encoding = depth_goal_encoder.extract_features(obsgoal_depth)
        depth_goal_encoding = depth_goal_encoder._avg_pooling(depth_goal_encoding)

        if depth_goal_encoder._global_params.include_top:
            depth_goal_encoding = depth_goal_encoding.flatten(start_dim=1)
            depth_goal_encoding = depth_goal_encoder._dropout(depth_goal_encoding)
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        depth_goal_encoding = compress_depth_goal_enc(depth_goal_encoding)
        if len(depth_goal_encoding.shape) == 2:
            depth_goal_encoding = depth_goal_encoding.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert depth_goal_encoding.shape[2] == depth_goal_encoding_size
        return depth_goal_encoding

    def obs_depth_encode(obs_depth: torch.tensor):
        # split the observation into context based on the context size
        # image size is [batch_size, 1*self.context_size, H, W]
        if( len(obs_depth.shape) == 3 ):
            obs_depth = obs_depth[None,...]
            
        obs_depth = torch.split(obs_depth, 1, dim=1)
        # image size is [batch_size*self.context_size, 1, H, W]
        obs_depth = torch.concat(obs_depth, dim=0)
        depth_obs_encoding = depth_obs_encoder.extract_features(obs_depth)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]  i.e., [B*Cont, 1280, 2, 2]
        depth_obs_encoding = depth_obs_encoder._avg_pooling(depth_obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if depth_obs_encoder._global_params.include_top:
            depth_obs_encoding = depth_obs_encoding.flatten(start_dim=1)
            depth_obs_encoding = depth_obs_encoder._dropout(depth_obs_encoding)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        depth_obs_encoding = compress_depth_obs_enc(depth_obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        depth_obs_encoding = depth_obs_encoding.reshape((context_size + 1, -1, obs_encoding_size))
        depth_obs_encoding = torch.transpose(depth_obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]
        return depth_obs_encoding

# def forward(      ):
    rgb_goal_encoding   = obsgoal_fused_rgb_encode(obs_rgb, goal_rgb)           # [B, 6, 512]
    rgb_obs_encoding    = obs_rgb_econde(obs_rgb)                               # [B, 1, 512]
    depth_goal_encoding = obsgoal_fused_depth_econde(obs_depth, goal_depth)     # [B, 1, 512]
    depth_obs_encoding  = obs_depth_encode(obs_depth)                           # [B, 6, 512]

    # concatenate the goal encoding to the observation encoding
    rgbd_tokens = torch.cat((rgb_obs_encoding, depth_obs_encoding, rgb_goal_encoding, depth_goal_encoding), dim=1)
    rgbd_final_repr = decoder(rgbd_tokens)      # transformer decoder based on self attentions
    # currently, the size is [batch_size, 32]

    dist_pred   = dist_predictor(rgbd_final_repr)
    action_pred = action_predictor(rgbd_final_repr)

    # augment outputs to match labels size-wise
    len_trajectory_pred = 5
    action_pred = action_pred.reshape( (action_pred.shape[0], len_trajectory_pred, num_action_params) )
    
    action_pred[:, :, :2] = torch.cumsum(
        action_pred[:, :, :2], dim=1
    )  # convert position deltas into waypoints
    if learn_angle:
        action_pred[:, :, 2:] = F.normalize(
            action_pred[:, :, 2:].clone(), dim=-1
        )  # normalize the angle prediction
    

###### self attention
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x













    dist_pred = self.dist_predictor(final_repr)
    action_pred = self.action_predictor(final_repr)

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
