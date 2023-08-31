"""
要调用训练的模型，使用的图片必须和训练用的图片的通道、大小数一致，进行transform处理方法也应该一致
使用gpu训练的模型在调用时图片和模型也应该增加.cuda()处理
"""
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
import torch.nn.functional as F

image_path = "E:\\img\\opencv\\num9.jpg"
image = Image.open(image_path)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])
image = transform(image)
# image=image.cuda()
print(image.shape)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model1_base = torch.load("all_paraments.pth")
# model1_base=model1_base.cuda()
print(model1_base)

image = torch.reshape(image, (1, 1, 28, 28))
model1_base.eval()
with torch.no_grad():
    output = model1_base(image)
print(output)
print(output.argmax(1))
