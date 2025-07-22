# Author: William Kandolo
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dense = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        h, _ = self.lstm(z)
        return self.dense(h)
