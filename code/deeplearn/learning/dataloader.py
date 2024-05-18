import torchvision
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

#数据集，批次大小，是否打乱，是否多线程（0表示只有一个主进程），是否放弃不成批的数据
test_loader =DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)