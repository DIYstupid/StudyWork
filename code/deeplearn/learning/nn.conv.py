import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64)

class G_module(nn.Module):

    def __init__(self) :
        super().__init__()
        # 一个卷积层
        self.conv1 = Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=0)
    def forward(self,x):
       x = self.conv1(x)
       return x

g_module = G_module()
# print(g_module)

writer = SummaryWriter("./logs")
step = 1
for data in dataloader:
    img,targets = data
    output = g_module(img)
    writer.add_image("input",img,step)
    writer.add_image("output",output,step)
    step += 1