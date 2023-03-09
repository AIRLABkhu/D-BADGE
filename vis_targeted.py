import os
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

import data, models

DEVICE = 'cuda:0'
batch_size = 128
dataset_name = 'cifar10'
resnet_depth = 18
exp_dir = f'./log/cifar10_resnet{resnet_depth}_1.0'

dataset = data.get_dataset(dataset_name, train=False, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=12)
categories = data.get_categories(dataset_name)

model = models.get_model(f'resnet{resnet_depth}').eval().to(DEVICE)
state_dict = torch.load(os.path.join(exp_dir, 'ckpt.pth'))['net']
model.load_state_dict(state_dict)

dirs = sorted(glob(os.path.join(exp_dir, 'target/*')))
uaps = [torch.load(os.path.join(d, 'ckpt.pth'))['uap'].to(DEVICE) for d in dirs]

for i, uap in enumerate(uaps):
    uap = uap.clone().detach()
    uap -= uap.min()
    uap /= uap.max()
    uap = resize(uap, size=(512, 512), interpolation=InterpolationMode.NEAREST)
    save_image(uap, f'temp/uap_{categories[i]}.png')

for i in range(3):
    img, _ = dataset[1251 + i]
    img = resize(img, size=(512, 512), interpolation=InterpolationMode.NEAREST)
    save_image(img, f'temp/img_{i}.png')
