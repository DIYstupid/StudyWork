import torch
from torch import nn

class G_Moudle(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input):
        output = input + 1
        return output
g_moudle = G_Moudle()
x = torch.tensor(1,0);
output = g_moudle