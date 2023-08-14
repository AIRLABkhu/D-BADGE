'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn

import torchvision.transforms as transforms

import os
import argparse
from pathlib import Path
from typing import OrderedDict
from tqdm import tqdm

import models, data

'''
python train_victim.py --device cuda:0 --lr 1.0E-3 --num-epochs 15  --model toy_mnist_t --dataset mnist   --tag mnist_toy_t
python train_victim.py --device cuda:7 --model vit_t --tag cifar10_vit_t_1.0
python train_victim.py --device cuda:7 --model vit_t --tag test --overwrite
'''
parser = argparse.ArgumentParser(description='PyTorch Victim Training')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--model', type=str, default='vgg19', choices=list(models.model_map.keys()))
parser.add_argument('--dataset', type=str, default='cifar10', choices=list(data.supported_datasets))
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--skip-norm', default=True, action='store_true')
parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--overwrite', default=False, action='store_true')
args = parser.parse_args()

checkpoint_dir = Path('./log').joinpath(args.tag)
if args.overwrite:
    os.system(f'rm -rf "{checkpoint_dir}"')
checkpoint_dir.mkdir(parents=True, exist_ok=True)
ckpt_filename = checkpoint_dir.joinpath('ckpt.pth')
if ckpt_filename.exists():
    if not args.overwrite:
        raise FileExistsError(f'Checkpoint file for tag="{args.tag}" already exists.')

# Data
print('==> Preparing data..')
# transform_train = [
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ]
# transform_train = [
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ]
transform_train = [
    transforms.RandAugment(2, 14),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
]
# transform_train = [  # ChatGPT recommentations
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.GaussianBlur(kernel_size=3),
#     transforms.ToTensor(),
# ]
transform_test = [
    transforms.ToTensor(),
]
if not args.skip_norm:
    norm_layer = transforms.Normalize(*data.get_mean_std(args.dataset))
    transform_train.append(norm_layer)
    transform_test.append(norm_layer)

trainset = data.get_dataset(dataset=args.dataset, train=True, transform=transforms.Compose(transform_train))
# trainset = data.get_dataset(dataset=args.dataset, train=False, transform=transforms.Compose(transform_train))
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = data.get_dataset(dataset=args.dataset, train=False, transform=transforms.Compose(transform_test))
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = data.get_categories(args.dataset)

# Model
print('==> Building model..')
model = models.get_model(args.model, num_classes=len(classes)).to(args.device)
if args.device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-4)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5.0E-4)

# scheduler = None
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5)

acc_list = []
best_acc = 0.0
best_epoch = -1

# Training
def train(epoch):
    model.train()
    train_loss = 0
    correct, total = 0, 0
    with tqdm(trainloader, desc='TRAIN', position=2, leave=False, dynamic_ncols=True) as progress_bar:
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            postfix_str = f'loss: {train_loss / (batch_idx + 1):.3f} | acc: {100 * correct / total:.3f}% ({correct}/{total})'
            progress_bar.set_postfix_str(postfix_str)
    return loss


def test(epoch):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad(), tqdm(testloader, desc=' EVAL', position=2, leave=False, dynamic_ncols=True) as progress_bar:
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = test_loss / (batch_idx + 1)
            acc = 100 * correct / total

            postfix_str = f'loss: {loss:.3f} | acc: {acc:.3f}% ({correct}/{total})'
            progress_bar.set_postfix_str(postfix_str)

    # Save checkpoint.
    global best_acc, best_epoch, acc_list
    acc_list.append(acc)
    if acc > best_acc:
        state = {
            'args': args,
            'model': args.model,
            'net': OrderedDict({key: val.cpu() for key, val in model.state_dict().items()}),
            'epoch': epoch,
            'acc': acc,
            'acc_list': acc_list,
        }
        best_acc = acc
        best_epoch = epoch
        torch.save(state, ckpt_filename)


start_epoch = 1
with tqdm(range(start_epoch, start_epoch + args.num_epochs), desc='EPOCH', position=1, leave=False, dynamic_ncols=True) as progress_bar:
    for epoch in progress_bar:
        loss = train(epoch)
        if scheduler is not None:
            scheduler.step(loss)
        test(epoch)
        
        progress_bar.set_postfix_str(f'best: {best_acc:.3f}% at {best_epoch} epoch')
    state = {
        'args': args,
        'model': args.model,
        'net': OrderedDict({key: val.cpu() for key, val in model.state_dict().items()}),
        'epoch': best_epoch,
        'acc': best_acc,
        'acc_list': acc_list,
    }
    torch.save(state, ckpt_filename)

print(f'The model got {best_acc:.3f}% at {best_epoch} epoch.')
