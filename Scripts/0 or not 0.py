import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from glob import glob
from torch.utils.data import DataLoader
class CustomDataset(Dataset):
    def __init__(self, xlsx_file, root_dir, transform=None):
        """
        Args:
            xlsx_file (string): 包含图片路径和标签的xlsx文件的路径。
            root_dir (string): 所有图片的目录路径。
            transform (callable, optional): 一个可选的变换，应用于样本。
        """
        self.labels_frame = pd.read_excel(xlsx_file)
        self.root_dir = root_dir
        self.transform = transform
        # 使用glob来获取所有图片路径
        self.image_paths = sorted(glob(self.root_dir))

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]  # 获取对应索引的图片路径
        image = Image.open(img_name)
        label = int(self.labels_frame.iloc[idx, 0])  # 标签在第一列

        if self.transform:
            image = self.transform(image)

        return image, label

# 设置数据转换
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 实例化数据集，注意路径用双反斜杠或原始字符串
dataset = CustomDataset(xlsx_file='label_0_or_not_0.xlsx',
                        root_dir=r'0_or_not_0\Datasets\imgs\*.png',
                        transform=data_transform)

# 打印一些样本来检查数据集
for i in range(len(dataset)):
    image, label = dataset[i]
    print(f'Image shape: {image.shape}, Label: {label}')
    if i == 3:  # 只打印前4个样本
        break

data_loader = DataLoader(dataset, batch_size=16, shuffle=True)



import torch
import torch.nn as nn

class BasicConv(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class VGG(nn.Module):

    def __init__(self, blocks, num_class=200):
        super().__init__()
        self.input_channels = 3
        self.conv1 = self._make_layers(64, blocks[0])
        self.conv2 = self._make_layers(128, blocks[1])
        self.conv3 = self._make_layers(256, blocks[2])
        self.conv4 = self._make_layers(512, blocks[3])
        self.conv5 = self._make_layers(512, blocks[4])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _make_layers(self, output_channels, layer_num):
        layers = []
        while layer_num:
            layers.append(
                BasicConv(
                    self.input_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                )
            )
            self.input_channels = output_channels
            layer_num -= 1
        layers.append(nn.MaxPool2d(2, stride=2))

        return nn.Sequential(*layers)

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# 定义设备，优先使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将模型转移到定义的设备上
model = VGG([2, 2, 4, 4, 4],2).to(device)
model.load_state_dict(torch.load("VGG_0_or_not_0_best.pth"))
# 定义损失函数
criterion = nn.CrossEntropyLoss()

from torch.optim import lr_scheduler
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.000001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
# 定义训练的轮数
num_epochs = 10
# 训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    # 使用 tqdm 包装你的数据加载器
    progress_bar = tqdm(enumerate(data_loader, 0), total=len(data_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')
    for i, data in progress_bar:
        # 获取输入
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        outputs = outputs.squeeze()  # 压缩输出以匹配标签的维度

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        running_loss += loss.item()
        accuracy = 100 * correct / total
        progress_bar.set_postfix({'loss': running_loss / (i + 1), 'accuracy': accuracy})


    # 打印每个epoch结束后的平均损失和准确率
    avg_loss = running_loss / len(data_loader)
    avg_accuracy = 100 * correct / total
    print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.2f}%')

print('Finished Training')

PATH = 'VGG_0_or_not_0_best.pth'
torch.save(model.state_dict(), PATH)