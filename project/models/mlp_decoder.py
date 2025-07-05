
# models/mlp_decoder.py

import torch
import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
