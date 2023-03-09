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
'''
parser = argparse.ArgumentParser(description='PyTorch Victim Training')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--model', type=str, default='vgg19', choices=list(models.model_map.keys()))
parser.add_argument('--dataset', type=str, default='cifar10', choices=list(data.supported_datasets))
parser.add_argument('--skip-norm', default=False, action='store_true')
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
transform_train = [
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]
transform_test = [
    transforms.ToTensor(),
]
if not args.skip_norm:
    norm_layer = transforms.Normalize(*data.get_mean_std(args.dataset))
    transform_train.append(norm_layer)
    transform_test.append(norm_layer)

trainset = data.get_dataset(dataset=args.dataset, train=True, transform=transforms.Compose(transform_train))
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = data.get_dataset(dataset=args.dataset, train=False, transform=transforms.Compose(transform_test))
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = data.get_categories(args.dataset)

# Model
print('==> Building model..')
model = models.get_model(args.model).to(args.device)
if args.device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
    global best_acc, best_epoch
    if acc > best_acc:
        state = {
            'args': args,
            'model': args.model,
            'net': OrderedDict({key: val.cpu() for key, val in model.state_dict().items()}),
            'epoch': epoch,
            'acc': acc,
        }
        best_acc = acc
        best_epoch = epoch
        torch.save(state, ckpt_filename)


start_epoch = 1
with tqdm(range(start_epoch, start_epoch + args.num_epochs), desc='EPOCH', position=1, leave=False, dynamic_ncols=True) as progress_bar:
    for epoch in progress_bar:
        train(epoch)
        test(epoch)
        scheduler.step()
        
        progress_bar.set_postfix_str(f'best: {best_acc:.3f}% at {best_epoch} epoch')

print(f'The model got {best_acc:.3f}% at {best_epoch} epoch.')
