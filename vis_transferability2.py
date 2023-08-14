import os
import argparse
import logging
from pathlib import Path
from glob import glob
from tqdm import tqdm

import seaborn as sn
import matplotlib.pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 13

import pandas as pd
pd.set_option('display.float_format', '{:.3f}%'.format)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import models, data
import utils

'''
python vis_transferability2.py  --device cuda:0 --num-baselines 5 --tag test --overwrite
python vis_transferability2.py  --device cuda:0 --tag default
'''
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Transferibility Visualization')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--num-baselines', type=int, default=5)
parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--checkpoints', type=str, nargs='*', default=['resnet18', 'vgg19', 'mobilenet_v2', 'vit', 'swin'])
parser.add_argument('--archnames', type=str, nargs='*', default=['RN18', 'VGG19', 'MBN_v2', 'ViT', 'Swin'])
parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--overwrite', default=False, action='store_true')

args = parser.parse_args()

DEVICE = args.device
torch.cuda.set_device(DEVICE)

# Paths
log_dir = Path('.').joinpath(args.log_dir, '_hypothesis_test', args.tag)
log_filename = log_dir.joinpath('test.log')
eval_filename = log_dir.joinpath('evaluation.pth')
make_eval_file = (not log_dir.exists()) or args.overwrite
if make_eval_file:
    os.system(f'rm -rf "{log_dir}"')
    log_dir.mkdir(parents=True)
    os.system(f'cp "{__file__}" "{log_dir}"')

# Logging
logger = logging.getLogger('TEST')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def _print(*args, sep=' ', tabs=0, tabc='  ', bullet=None, number=None):
    str_args = [str(arg) for arg in args]
    line = sep.join(str_args)
    if bullet:
        if bullet is True:
            bullet = '•'
        line = f'{bullet} {line}'
    if number:
        line = f'{number}. {line}'
    if tabs is not None:
        line = (tabc * tabs) + line
    logger.info(line)
    print(line)

_print(args)

# Checkpoints
archdirs = [f'./log/cifar10_{arch}_1.0' for arch in args.checkpoints]
arch_ckpts = [torch.load(os.path.join(d, 'ckpt.pth')) for d in archdirs]
nets = []
for ckpt in arch_ckpts:
    net = models.get_model(ckpt['model']).to(DEVICE).eval()
    net.load_state_dict(ckpt['net'])
    nets.append(net)
    
uaps = []
for archdir in archdirs:
    filenames = sorted(glob(os.path.join(archdir, '_baselines', '*', 'ckpt.pth')))
    for fn in filenames[:args.num_baselines]:
        uaps.append(torch.load(fn)['uap'])
uaps = torch.cat(uaps, dim=0).to(DEVICE)

# Evaluation
_print('Loading dataset...')
transform = [
    transforms.ToTensor(),
]
testset = data.get_dataset('cifar10', train=False, transform=transforms.Compose(transform))
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

@torch.no_grad()
def eval_uap(victim, uap, desc='EVAL'):
    correct_cln, correct_adv, fooled, total = 0, 0, 0, 0
    with tqdm(test_loader, desc=desc, position=3, leave=False) as test_bar:
        for inputs, targets in test_bar:
            batch_size = inputs.size(0)
            inputs = inputs.to(DEVICE)
            
            inputs_cln = inputs
            inputs_adv = torch.add(inputs_cln, uap.unsqueeze(0))
            inputs_all = torch.cat([inputs_cln, inputs_adv], dim=0)
            inputs_all = torch.clamp(inputs_all, 
                                     min=inputs.amin(dim=(0, 2, 3), keepdim=True), 
                                     max=inputs.amax(dim=(0, 2, 3), keepdim=True))
            
            outputs_all = victim(inputs_all)
            preds_all = outputs_all.argmax(dim=1).cpu()
            preds_cln, preds_adv = torch.split(preds_all, batch_size, dim=0)
            
            correct_cln += (preds_cln == targets).sum().item()
            correct_adv += (preds_adv == targets).sum().item()
            fooled += (preds_cln != preds_adv).sum().item()
            total += batch_size
            
            fr = 100 * fooled / total
            acc_cln = 100 * correct_cln / total
            acc_adv = 100 * correct_adv / total
            
            test_bar.set_postfix_str(f'{fr=:.3f}% | {acc_adv=:.3f}% | {acc_cln=:.3f}%')
    return fr

fr_grids = []
for net in tqdm(nets, desc='NET', position=1, leave=False):
    fr_grid = []
    for uap in tqdm(uaps, desc='UAP', position=2, leave=False):
        fr = eval_uap(net, uap)
        fr_grid.append(fr)
    fr_grid = torch.tensor(fr_grid).view(-1, args.num_baselines)
    fr_grids.append(fr_grid.mean(dim=1))
fr_grids = torch.stack(fr_grids, dim=0)

print(fr_grids)

