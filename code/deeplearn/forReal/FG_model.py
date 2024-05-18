import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Sigmoid, Softmax


class G_mod(nn.Module):
    def __init__(self):
        super(G_mod,self).__init__()
        self.feature = Sequential(
            Conv2d(5, 16, 3, padding=1),
            ReLU(),
            MaxPool2d(2,2),
            Conv2d(16, 32, 3, padding=1),
            ReLU(),
            MaxPool2d(2,2)
        )
        self.classif =Sequential(
            Linear(32 * 4 * 4,128),
            ReLU(),
            Linear(128,2),
            #Sigmoid()
            Softmax(1)
        )
    def forward(self,x):
        x = self.feature(x)

        x = x.view(-1,32 * 4 * 4)

        x = self.classif(x)
        return x
if __name__ == '__main__':
    a = torch.ones([1,5, 17,16])
    model = G_mod()
    b = model(a)
    print(b[0].argmax().item())
    print(b)
    print(b.shape)