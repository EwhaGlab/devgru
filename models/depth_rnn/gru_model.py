import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size,  seq_len):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        # (512 * 2 = 1024) --> 512 --> 256 --> 128 --> 64 --> 32
        #output_layers = [256, 128, 64, 32]
        output_layers = [int(input_size/2), int(input_size/4), int(input_size/8), int(input_size/16)]
        #self.output_layers = nn.ModuleList([nn.Linear(seq_len*input_size, input_size)])
        self.output_layers = nn.ModuleList([nn.Linear(2 * input_size, input_size)])
        self.output_layers.append(nn.Linear(input_size, output_layers[0])) # nn.Linear(512, 256)
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        (B, S, num_feat) = x.shape  # batch_size, seq_len, num features
        x, _ = self.gru(x)
        #x = x.reshape(x.shape[0], -1)
        last_two = x[:, -2:, :]
        x = last_two.reshape(B,-1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
        
        
    
    
