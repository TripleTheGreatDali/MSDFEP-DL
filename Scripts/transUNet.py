import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import matplotlib.pyplot as plt

# -------------------------------------------------------------------      Model        ----------------------------------------------------
img_size = 128
BATCH_SIZE = 16
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=BATCH_SIZE, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=img_size, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args(args=[])
if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
PATH = "TransUnet_gray_best_2024_07.pth"
net.load_state_dict(torch.load(PATH))
net
# -------------------------------------------------------------------      Dataset        ----------------------------------------------------

import glob
import numpy as np
from torchvision import transforms
from torch.utils import data
from PIL import Image
# Train Data
# dsb2018_96/images\*.png
# dsb2018_96/masks/0\*.png

Train_images_Path = f"transUnet\inputs_org_size_corp_img/*.png"
Train_masks_Path = f"transUnet\inputs_org_size_corp_mask/*.png"
all_pics = glob.glob(r'{}'.format(Train_images_Path))
all_masks = glob.glob(r'{}'.format(Train_masks_Path))
images = [p for p in all_pics]
masks = [p for p in all_masks]

np.random.seed(2021)
index = np.random.permutation(len(images))
images = np.array(images)[index]
masks = np.array(masks)[index]

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

class Portrait_dataset(data.Dataset):
    def __init__(self, img_paths, anno_paths):
        self.imgs = img_paths
        self.annos = anno_paths

    def __getitem__(self, index):
        img = self.imgs[index]
        anno = self.annos[index]

        # pil_img = Image.open(img).convert('RGB')  # 确保图像是RGB
        pil_img = Image.open(img).convert('L')  # 确保图像是RGB
        img_tensor = transform(pil_img)

        pil_anno = Image.open(anno).convert('L')  # 如果mask是单通道灰度图，可以使用 'L'
        anno_tensor = transform(pil_anno)
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long)
        anno_tensor[anno_tensor > 0] = 1

        return img_tensor, anno_tensor

    def __len__(self):
        return len(self.imgs)

train_dataset = Portrait_dataset(images, masks)
train_dl = data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
)
imgs_batch, annos_batch = next(iter(train_dl))


# -------------------------------------------------------------------      train        ----------------------------------------------------
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
from torch.optim import lr_scheduler

optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
from tqdm import tqdm

epochs = 100
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    total_correct = 0
    num_batches = 0
    total_running_loss = 0
    total = 0

    net.train()
    # 使用 tqdm 包装你的数据加载器
    progress_bar = tqdm(enumerate(train_dl, 0), total=len(train_dl), desc=f'Epoch {epoch + 1}/{epochs}')
    for i, data in progress_bar:
        x, y = data
        # print(x.size())
        # break
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = net(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            # 每个批次的
            correct_batch = (y_pred == y).float()
            accuracy = correct_batch.sum() / y.numel()

            # 总共的
            total_correct += correct_batch.sum()
            total += y.numel()
            total_running_loss += loss.item()
            num_batches += 1
        progress_bar.set_postfix({'loss': loss.item(), 'accuracy': accuracy})

    exp_lr_scheduler.step()
    print("total_loss:", total_running_loss / num_batches, "total_accuracy:", total_correct / total)
    # break

PATH = './TransUnet_gray_best_2024_07.pth'
torch.save(net.state_dict(), PATH)


































