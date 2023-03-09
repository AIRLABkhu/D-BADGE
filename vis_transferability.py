import os
import argparse
import logging
from pathlib import Path
import itertools
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
python hyp_test.py  --device cuda:0 --tag default
'''
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Transferibility Visualization')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--num-baselines', type=int, default=5)
parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--checkpoints', type=str, nargs='*', default=['cifar10_resnet18_1.0',
                                                                   'cifar10_resnet20_1.0',
                                                                   'cifar10_vgg19_1.0',
                                                                   'cifar10_mobilenet_v2_1.0',
                                                                   'cifar10_resnext29_2x64d_1.0',
])
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
ckpt_dirs = sorted([os.path.join('.', args.log_dir, d.strip()) for d in args.checkpoints])

_print('Loading checkpoints...')
checkpoints = [torch.load(os.path.join(d, 'ckpt.pth')) for d in ckpt_dirs]
arch_names = [ckpt['model'] for ckpt in checkpoints]

_print('Architectures:')
for arch_name in arch_names:
    _print(arch_name, tabs=1, bullet=True)
max_len_arch_name = max(len(n) for n in arch_names)

model_pool = [models.get_model(ckpt['model']) for ckpt in checkpoints]
for model, ckpt in zip(model_pool, checkpoints):
    model.load_state_dict(ckpt['net'])
    model.eval()

uap_pool = torch.cat([
    torch.load(d)['uap']
    for d in sorted(itertools.chain([
        os.path.join(d, '_baselines', f'{i:02d}', 'ckpt.pth') 
        for i in range(args.num_baselines) for d in ckpt_dirs]))
], dim=0).to(DEVICE)
noise_pool = utils.shuffle_pixels(uap_pool)
uap_pool = [uap for uap in uap_pool.reshape(-1, 5, *uap_pool.shape[1:])]
noise_pool = [uap for uap in noise_pool.reshape(-1, 5, *noise_pool.shape[1:])]

_print('Loading dataset...')
transform = [
    transforms.ToTensor(),
    # transforms.Normalize(*data.get_mean_std(dataset)),
]
testset = data.get_dataset('cifar10', train=False, transform=transforms.Compose(transform))
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

@torch.no_grad()
def eval_uap(victim, uap, desc='EVAL'):
    correct_cln, correct_adv, fooled, total = 0, 0, 0, 0
    with tqdm(test_loader, desc=desc, position=2, leave=False) as test_bar:
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

if make_eval_file:
    _print('Building baseline with noise...')
    noise_fr = []
    mean_noise_fr = []
    with tqdm(zip(model_pool, noise_pool, arch_names), desc='NOISE', position=1, leave=False) as victim_bar:
        for victim, noises, arch_name in victim_bar:
            victim = victim.to(DEVICE)
            fr_noise_list = []
            for idx, noise in enumerate(noises, start=1):
                victim_bar.set_postfix_str(f'{arch_name} ({idx}/{len(noises)})')
                fr_noise = eval_uap(victim, noise)
                fr_noise_list.append(fr_noise)
                
            noise_fr.append(fr_noise_list)
            victim = victim.cpu()
    noise_fr = torch.tensor(noise_fr)
    mean_noise_fr = noise_fr.mean(dim=1)
    print()

    _print('Baselines (noise):')
    for name, fr_noise in zip(arch_names, mean_noise_fr):
        _print(f'{name:<{max_len_arch_name}}: {fr_noise:.3f}%', tabs=1, bullet=True)

    _print('Evaluating Combinations...')
    uap_pool = torch.cat(uap_pool, dim=0)
    fr_grid = torch.zeros(len(model_pool), uap_pool.size(0))
    with tqdm(zip(model_pool, arch_names), desc='UAP', position=1, leave=False) as victim_bar:
        for victim_idx, (victim, arch_name) in enumerate(victim_bar):
            victim = victim.to(DEVICE)
            for uap_idx, uap in enumerate(uap_pool, start=1):
                victim_bar.set_postfix_str(f'{arch_name} ({uap_idx}/{len(uap_pool)})')
                fr = eval_uap(victim, uap)
                fr_grid[victim_idx, uap_idx - 1] = fr
            victim = victim.cpu()
    print()

    evaluation = {
        'arch_names': arch_names,
        'noise_fr': noise_fr,
        'fr_grid': fr_grid,
    }
    torch.save(evaluation, eval_filename)
else:
    print()
    print('Loading evaluation result...')
    uap_pool = torch.cat(uap_pool, dim=0)
    evaluation = torch.load(eval_filename)
    noise_fr = evaluation['noise_fr']
    mean_noise_fr = noise_fr.mean(dim=1)
    fr_grid = evaluation['fr_grid']
    print()
    
num_victims = fr_grid.size(0)
num_uaps = fr_grid.size(1) // num_victims
mean_fr_grid = fr_grid.view(num_victims, num_victims, num_victims).mean(dim=-1)

import matplotlib.pyplot as plt
plt.tight_layout()
ax = plt.subplot(111)
ax.matshow(mean_fr_grid, cmap=plt.cm.Blues)
threshold = mean_fr_grid.min() + (mean_fr_grid.max() - mean_fr_grid.min()) * 0.7
for i in range(mean_fr_grid.shape[0]):
    for j in range(mean_fr_grid.shape[1]):
        val = mean_fr_grid[i, j]
        color = 'white' if val >= threshold else 'black'
        ax.text(x=j, y=i, s=f'{val:.2f}%', va='center', ha='center', color=color)
ax.xaxis.set_label_text('Target victim')
ax.yaxis.set_label_text('Source victim')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks(range(num_victims))
ax.set_yticks(range(num_victims))
arch_names = sorted(['RN18', 'RN20', 'VGG19', 'MBN_V2', 'RNX29'])
ax.set_xticklabels(arch_names, rotation=30)
ax.set_yticklabels(arch_names, rotation=30)

plt.savefig('temp/temp.png')
plt.savefig('temp/temp.pdf')

def print_as_df(data: torch.Tensor, title=None, index=None, columns=None, print_fn=None, float_fmt=None, newline=True):
    if index is None:
        index = lambda i: f'{i}'
    if columns is None:
        columns = lambda i: f'{i}'
    df = pd.DataFrame(data).rename(
        index=index, columns = columns
    )
    
    if print_fn is None:
        print_fn = _print if make_eval_file else print
    if float_fmt is None:
        float_fmt = '{:.3f}%'
    pd.set_option('display.float_format', float_fmt.format)
    if title is not None:
        print(title)
    print_fn(df)
    if newline:
        print()
    pd.set_option('display.float_format', '{:.3f}%'.format)
    
print_as_df(mean_fr_grid, title='Mean Fooling rates (Architecture-Perturbation):', 
            index=lambda i: f'{arch_names[i]:<{max_len_arch_name}} (A{i})',
            columns=lambda i: f'A{i}')
