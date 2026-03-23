import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size,  seq_len, final_dim=128, p_drop=0.1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        # (512 * 2 = 1024) --> 512 --> 256 --> 128   # --> 64 --> 32
        # output layer becomes [256,128,128,128] if the final_dim is 128
        # output layer becomes [256,128,64,32] if the final_dim is 32
        assert (input_size % final_dim == 0)
        output_layers = []
        dim = input_size
        while dim > final_dim:
            dim = max(final_dim, dim // 2)
            output_layers.append(dim)
        while len(output_layers) < 4:
            output_layers.append(final_dim)

        self.output_layers = nn.ModuleList([nn.Linear(2 * hidden_size, hidden_size)])
        self.output_layers.append(nn.Linear(hidden_size, output_layers[0])) # nn.Linear(512, 256)
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

        self.dropout = nn.Dropout(p_drop)
    def forward(self, x):
        (B, S, num_feat) = x.shape  # batch_size, seq_len, num features
        x, _ = self.gru(x)
        #x = x.reshape(x.shape[0], -1)
        last_two = x[:, -2:, :]
        z = last_two.reshape(B,-1)

        for i, layer in enumerate(self.output_layers):
            z = layer(z)
            if i < len(self.output_layers) - 1:
                z = F.relu(z)
                z = self.dropout(z)
        return z  # [B, final_dim]

        # for i in range(len(self.output_layers)):
        #     x = self.output_layers[i](x)
        #     x = F.relu(x)
        # return x
        
        
    
    
