import torch
import numpy as np


class TimeEncode(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        batch_size, seq_len = t.shape
        t = t.unsqueeze(dim=2).view(batch_size * seq_len, -1)
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))
        output = output.reshape(batch_size, seq_len, -1)
        return output
