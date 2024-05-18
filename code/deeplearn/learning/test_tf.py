from PIL import Image
from torchvision import transforms
from  torch.utils.tensorboard import SummaryWriter
#python的用法 - 》 tensor数据类性
# 通过transforms.ToTensor 解决两个问题
# transform how to use
# what it tensor
writer = SummaryWriter("logs")
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

img_norm = transforms.Normalize()
