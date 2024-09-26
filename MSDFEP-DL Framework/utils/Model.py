import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from .networks.vit_seg_modeling import VisionTransformer as ViT_seg
from .networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import matplotlib.pyplot as plt

import torch.nn as nn
from PIL import Image
from glob import glob
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

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

def Transunet():
    img_size = 128
    BATCH_SIZE = 8
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
    return net