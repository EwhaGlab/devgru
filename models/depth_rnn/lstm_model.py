import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        output_layers = [256, 128, 64, 32]
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*input_size, input_size)])
        self.output_layers.append(nn.Linear(input_size, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))
        #self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
        
        
    
    
