import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda")
train_data = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                        ]))
test_data = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ]))

# 查看数据集的大小
test_data_size = len(test_data)
train_data_size = len(train_data)
print("训练集测试长度：{}".format(test_data_size))
print("测试集测试长度：{}".format(train_data_size))

train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)


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


# 创建模型
class_model = Net()
class_model = class_model.cuda()

# 创建损失函数
loss_nn = nn.CrossEntropyLoss()
loss_nn = loss_nn.cuda()

# 优化器
optim = torch.optim.SGD(class_model.parameters(), lr=0.01)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练的轮数
epoch = 15
# 总体的正确率
total_accurrcy = 0

for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = class_model(imgs)
        loss = loss_nn(outputs, targets)

        # 优化器调参数
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{},loss:{}".format(total_train_step, loss.item()))

    # 测试步骤
    total_test_loss = 0
    with torch.no_grad():  # 测试集不需要梯度进行测试和优化
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = class_model(imgs)
            loss = loss_nn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accurrcy = (outputs.argmax(1) == targets).sum()
            total_accurrcy = total_accurrcy + accurrcy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集的正确率：{}".format(total_accurrcy / test_data_size))
    total_test_step = total_test_step + 1

# 训练模型保存
torch.save(class_model, "try_model1.pth")
