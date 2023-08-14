'''Train CIFAR10 with PyTorch.'''
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import random
from time import time
import numpy as np

import cfg
import models, data
import optimizers, schedulers
import loss_func

SUPPORTED_REGULATIONS = ['clamp', 'inf', 'fro']

'''
python train_attack.py --device cuda:0 -c cifar10-optim/adam --checkpoint cifar10_resnext29_2x64d_1.0 --tag exp_others
python train_attack.py --device cuda:6 -c cifar10-optim/adam --checkpoint test --tag baseline_00

python train_attack.py --device cuda:0 -c cifar10-optim/adam --checkpoint cifar10_vit_1.0 --tag baseline_00
'''
parser = argparse.ArgumentParser(description='PyTorch UAP Training')
parser.add_argument('--config', '-c', type=str, nargs='*', default=[])

parser.add_argument('--device', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--benchmark', action='store_true')

parser.add_argument('--epochs', type=int)
parser.add_argument('--beta', type=float)
parser.add_argument('--gamma', type=float)
parser.add_argument('--learning-rate', '-lr', type=float)

parser.add_argument('--batch-size', type=int)
parser.add_argument('--eval-batch-size', type=int)
parser.add_argument('--images-per-class', '-ipc', type=int)
parser.add_argument('--eval-images-per-class', '-eipc', type=int)
parser.add_argument('--accumulation', type=int)
parser.add_argument('--max-iters', type=int)
parser.add_argument('--sliding-window-batch', '-swb', action='store_true')
parser.add_argument('--augmentations', '-aug', type=str, nargs='*')

parser.add_argument('--target', type=int)
parser.add_argument('--budget', type=float)
parser.add_argument('--regulation', type=str, choices=SUPPORTED_REGULATIONS)
parser.add_argument('--eval-step-size', type=int)

parser.add_argument('--useeye', action='store_true')
parser.add_argument('--loss-func', type=str, choices=list(loss_func.loss_functions.keys()))
parser.add_argument('--use-logits', action='store_true')

parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--tag', type=str, required=True)

args = vars(parser.parse_args())
args = cfg.load(*args['config'], cfg_dir='train_attack', args=args)

# Paths
ckpt_dir = Path('./log').joinpath(args.checkpoint)
ckpt_filename = ckpt_dir.joinpath('ckpt.pth')
exp_dir = ckpt_dir.joinpath(args.tag)
code_dir = exp_dir.joinpath('codes')
aug_dir = code_dir.joinpath('augmentations')
vis_dir = exp_dir.joinpath('visualization')
log_filename = exp_dir.joinpath('train.log')
uap_filename = exp_dir.joinpath('ckpt.pth')
best_filename = vis_dir.joinpath('adversarial_examples.png')
if exp_dir.exists():
    os.system(f'rm -rf "{exp_dir}"')
code_dir.mkdir(parents=True, exist_ok=False)
vis_dir.mkdir(parents=True, exist_ok=True)

# Logging
logger = logging.getLogger('TRAIN')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def _print(*args, sep=' '):
    str_args = [str(arg) for arg in args]
    line = sep.join(str_args)
    logger.info(line)
    print(line)

_print('Algorithm: SPSA')
_print(args)

writer = SummaryWriter(exp_dir)
os.system(f'cp "{__file__}" "{code_dir}"')
    
augmentation_transform, aug_filenames = data.augmentations.build_transform(*args.augmentations)
if len(aug_filenames) > 0:
    aug_dir.mkdir(parents=True, exist_ok=True)
for filename in aug_filenames:
    src_filename = os.path.join('./cfg/aug', filename)
    dst_filename = aug_dir.joinpath(filename)
    os.system(f'cp "{src_filename}" "{dst_filename}"')
_print('Directories and loggers set')

# Checkpoint
checkpoint = torch.load(ckpt_filename)

# Seed
DEVICE = args.device
if args.seed is None:
    seed = torch.initial_seed() % 2 ** 32
else:
    seed = cfg.__SEEDS__[args.seed] % 2 ** 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if DEVICE.lower().startswith('cuda'):
    torch.cuda.set_device(DEVICE)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = args.deterministic

def seed_worker(worker_id):
    np.random.seed(seed)
    random.seed(seed)

g = torch.Generator()
g.manual_seed(0)

# Dataset
print('==> PREPAIR DATA')
dataset = checkpoint['args'].dataset
transform = [
    transforms.ToTensor(),
]
if 'skip_norm' not in checkpoint:
    checkpoint['args'].skip_norm = True
if not checkpoint['args'].skip_norm:
    transform.append(transforms.Normalize(*data.get_mean_std(dataset)))
trainset = data.get_dataset(dataset, train=True, transform=transforms.Compose(augmentation_transform + transform))
testset = data.get_dataset(dataset, train=False, transform=transforms.Compose(transform))

if args.images_per_class != -1:
    indices = []
    for i in range(10):
        indices_i = [j for j in range(len(trainset)) if trainset[j][1] == i]
        indices.append(indices_i[:args.images_per_class])
    indices = [i for sublist in indices for i in sublist]
    trainset = Subset(trainset, indices)
if args.eval_images_per_class != -1:
    indices = []
    for i in range(10):
        indices_i = [j for j in range(len(testset)) if testset[j][1] == i]
        indices.append(indices_i[:args.eval_images_per_class])
    indices = [i for sublist in indices for i in sublist]
    testset = Subset(testset, indices)

loader_kwargs = dict(num_workers=4, 
                     worker_init_fn=seed_worker, 
                     generator=g)
trainloader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size, **loader_kwargs)
testloader = DataLoader(testset, shuffle=False, batch_size=args.eval_batch_size, **loader_kwargs)
classes = data.get_categories(dataset)

sample_indices = torch.tensor([5588, 6430, 4185]).long()
samples = torch.cat([testset[i][0].unsqueeze(0) for i in sample_indices], dim=0).to(DEVICE)
num_channels, input_height, input_width = samples[0].shape

# Model
print('==> BUILD MODEL')
model = models.get_model(checkpoint['model'], num_classes=len(classes)).eval().to(DEVICE)
model.load_state_dict(checkpoint['net'])
criterion = loss_func.get_loss_function(args.loss_func)

def evaluation():
    print('==> EVALUATE VICTIM')
    test_loss = 0
    correct, total = 0, 0
    with torch.no_grad(), tqdm(testloader, 'VICTIM', dynamic_ncols=True) as progress_bar:
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            mean_loss = test_loss / (batch_idx + 1)
            acc = 100 * correct / total

            postfix_str = f'loss: {mean_loss:.3f} | acc: {acc:.3f} ({correct}/{total})'
            progress_bar.set_postfix_str(postfix_str)
    _print(f'Victim ({checkpoint["model"]}) accuracy: {acc:.04f}%')
    print()

@torch.no_grad()
def batch_loss():
    def regulate_perturbation(uap: torch.Tensor, budget: float=10.0, mode: str='inf'):
        if mode not in SUPPORTED_REGULATIONS:
            raise NotImplementedError(f'"{mode}" is not a supported regulation mode. ' 
                                      f'Please select one of {", ".join(SUPPORTED_REGULATIONS)}.')
        
        budget = budget / 255.0
        if mode == 'clamp':
            return uap.clamp_(min=-budget, max=budget)
        elif mode in {'inf', 'fro'}:
            # Regulation on GAP
            norm_val = uap.norm(p={'inf': np.inf, 'fro': 2}[mode], dim=(1, 2, 3), keepdim=True) 
            ones = torch.ones_like(norm_val)
            scaler = torch.min(ones, budget / norm_val)
            return uap.multiply_(scaler)
        else:
            assert mode in SUPPORTED_REGULATIONS

    uap = torch.zeros(1, num_channels, input_height, input_width).to(DEVICE)
    optimizer = optimizers.get_optimizer(args.optimizer.name, params=[uap], lr=args.learning_rate, **args.optimizer.kwargs)
    lr_scheduler = schedulers.get_scheduler(args.lr_scheduler.name, optimizer=optimizer, **args.lr_scheduler.kwargs)
    beta_scheduler = schedulers.get_scheduler(args.beta_scheduler.name, value=args.beta, **args.beta_scheduler.kwargs)

    print('==> TRAIN PERTURBATION')
    training_time = 0
    max_iters = args.max_iters if args.max_iters != -1 else len(trainloader)
    with tqdm(range(1, args.epochs + 1), desc='EPOCH', position=1, leave=False, dynamic_ncols=True) as epoch_bar:
        best_uap = uap.clone().detach().cpu()
        best_fr = -1.0E-8
        best_epoch = 0
        fr_list = []
        for epoch in epoch_bar:
            logger.info(f'[EPOCH {epoch}]')

            with tqdm(trainloader, desc='TRAIN', position=2, leave=False, total=max_iters, dynamic_ncols=True) as train_bar:
                total, correct = 0, 0
                accm_outputs_neg, accm_outputs_pos = [], [] 
                accm_onehots = []  # batch accumulation
                start_time = time()
                for batch_idx, (inputs, targets) in enumerate(train_bar):
                    if batch_idx == max_iters:
                        break
                    batch_size = inputs.size(0)
                    
                    indices = torch.randperm(batch_size)
                    inputs = inputs[indices]
                    targets = targets[indices]
                    
                    beta = beta_scheduler.value
                    gamma = beta * 2.0
                    # BEGIN (SPSA specific region)
                    if args.useeye:
                        u = torch.eye(n=uap.size(2), device=DEVICE).float()
                        p = torch.randint(low=0, high=u.size(0), size=(3,))
                        u = torch.stack([torch.roll(u, shifts=(pn,), dims=(0,)) for pn in p], dim=0).unsqueeze(0)
                        # u = torch.clamp(u + 1.0E-8, 0, 1)
                        u = u * 2.0 - 1.0
                        # u *= torch.randn_like(u).sign()
                    else:
                        u = (torch.rand_like(uap, device=DEVICE) > 0.5).float() * 2.0 - 1.0
                    u_beta = u * beta
                    # END
                    onehots = nn.functional.one_hot(targets, len(classes))
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    
                    # BEGIN (SPSA specific region)
                    uap_neg = torch.sub(uap, u_beta)
                    uap_pos = torch.add(uap, u_beta)
                    uap_neg = regulate_perturbation(uap_neg, budget=args.budget, mode=args.regulation)
                    uap_pos = regulate_perturbation(uap_pos, budget=args.budget, mode=args.regulation)

                    inputs_adv_neg = torch.add(inputs, uap_neg)
                    inputs_adv_pos = torch.add(inputs, uap_pos)
                    inputs_all = torch.cat([inputs_adv_neg, inputs_adv_pos], dim=0)
                    inputs_all = torch.clamp(inputs_all, 
                                             min=inputs.amin(dim=(0, 2, 3), keepdim=True), 
                                             max=inputs.amax(dim=(0, 2, 3), keepdim=True))
                    # END

                    outputs_all = model(inputs_all)
                    outputs_neg, outputs_pos = outputs_all.split(split_size=batch_size, dim=0)
                    predicted_pos = outputs_pos.argmax(dim=1)

                    total += targets.size(0)
                    correct += (predicted_pos == targets).sum().item()

                    accm_outputs_neg.append(outputs_neg)
                    accm_outputs_pos.append(outputs_pos)
                    if args.target == -1:
                        accm_onehots.append(onehots.float().to(DEVICE))
                    else:
                        _targets = torch.ones_like(targets) * args.target
                        _onehots = nn.functional.one_hot(_targets, len(classes)).float().to(DEVICE)
                        accm_onehots.append(_onehots)
                    if len(accm_outputs_neg) == args.accumulation:
                        outputs_neg = torch.cat(accm_outputs_neg, dim=0)
                        outputs_pos = torch.cat(accm_outputs_pos, dim=0)
                        onehots = torch.cat(accm_onehots, dim=0)
                        
                        if not args.use_logits:
                            outputs_neg = nn.functional.one_hot(outputs_neg.argmax(dim=1), len(classes)).float().to(DEVICE)
                            outputs_pos = nn.functional.one_hot(outputs_pos.argmax(dim=1), len(classes)).float().to(DEVICE)

                        fool_neg = criterion(outputs_neg, onehots)
                        fool_pos = criterion(outputs_pos, onehots)
                        if args.target == -1:
                            fool_neg = 1 - fool_neg
                            fool_pos = 1 - fool_pos

                        # BEGIN
                        uap.grad = ((fool_neg - fool_pos) / gamma) / u
                        # END
                        optimizer.step()
                        uap = regulate_perturbation(uap, budget=args.budget, mode=args.regulation)
                        
                        if args.sliding_window_batch:
                            accm_outputs_neg = accm_outputs_neg[1:]
                            accm_outputs_pos = accm_outputs_pos[1:]
                            accm_onehots = accm_onehots[1:]
                        else:
                            accm_outputs_neg.clear()
                            accm_outputs_pos.clear()
                            accm_onehots.clear()

                    if len(accm_outputs_neg) > args.accumulation:
                        accm_outputs_neg = accm_outputs_neg[1:]
                        accm_outputs_pos = accm_outputs_pos[1:]
                        accm_onehots = accm_onehots[1:]
                    
                    acc = 100 * correct / total
                    train_bar.set_postfix_str(f'acc: {acc:.3f}')
                logger.info(f'Train accuracy: {acc:.3}%')
                writer.add_scalar("train/accuracy", acc, epoch)
                
                training_time += time() - start_time

            if epoch % args.eval_step_size == 0 or epoch == 1:
                with tqdm(testloader, desc=' EVAL', position=2, leave=False, dynamic_ncols=True) as eval_bar:
                    total, correct, fooled = 0, 0, 0
                    all_predictions = []
                    for inputs, targets in eval_bar:
                        batch_size = inputs.size(0)
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        
                        inputs_adv_0 = torch.add(inputs, uap)
                        inputs_all = torch.cat([inputs, inputs_adv_0], dim=0)
                        inputs_all = torch.clamp(inputs_all, 
                                                 min=inputs.amin(dim=(0, 2, 3), keepdim=True), 
                                                 max=inputs.amax(dim=(0, 2, 3), keepdim=True))

                        outputs_all = model(inputs_all)
                        predicted_all = outputs_all.argmax(dim=1)
                        predicted_c, predicted_0 = predicted_all.split(split_size=batch_size, dim=0)
                        all_predictions.append(predicted_0)

                        total += targets.size(0)
                        correct += (predicted_0 == targets).sum().item()
                        if args.target == -1:
                            fooled += (predicted_0 != predicted_c).sum().item()
                        else:
                            _target = torch.zeros_like(predicted_0).fill_(args.target)
                            fooled += (predicted_0 == _target).sum().item()

                        acc = 100 * correct / total
                        fr = 100 * fooled / total
                        
                        eval_bar.set_postfix_str(f'acc: {acc:.3f} | fr: {fr:.3}')
                    all_predictions = torch.cat(all_predictions, dim=0)
                    
                    fr_list.append(fr)
                    if fr >= best_fr:
                        best_uap = uap.clone().detach().cpu()
                        best_fr = fr
                        best_epoch = epoch
                        
                        minmax_uap = uap - uap.min()
                        minmax_uap = minmax_uap / minmax_uap.max()
                        examples = torch.add(samples, uap)
                        examples = torch.clamp(examples, 
                                            min=examples.amin(dim=(0, 2, 3), keepdim=True), 
                                            max=examples.amax(dim=(0, 2, 3), keepdim=True))
                        examples = torch.cat([torch.zeros_like(uap), samples, minmax_uap, examples], dim=0)
                        examples = torchvision.utils.make_grid(examples, nrow=int(round(examples.size(0) / 2)), padding=1)
                        torchvision.utils.save_image(examples, best_filename)
                        
                        uap_checkpoint = {
                            'args': args,
                            'epoch': epoch,
                            'fr_list': fr_list,
                            'fr': fr,
                            'time': training_time,
                            'uap': uap.cpu().clone().detach(),
                        }
                        torch.save(uap_checkpoint, uap_filename)
                        
                    logger.info(f'Evaluation fooling-rate: {fr:.3}%')
                    logger.info(f'Evaluation best fr:      {best_fr:.3}% at {best_epoch} epoch')
                    logger.info(f'Evaluation accuracy:     {acc:.3}%')
                    writer.add_scalar('eval/fooling-rate', fr, epoch)
                    writer.add_scalar('eval/best fr', best_fr, epoch)
                    writer.add_scalar('eval/accuracy', acc, epoch)
                    writer.add_histogram('eval/prediction', all_predictions, epoch)
                    
            l1_norm = uap.norm(p=1)
            l2_norm = uap.norm(p=2)
            inf_norm = uap.norm(p=np.inf)
            
            epoch_bar.set_postfix_str(f'best_fr: {best_fr:.3} at {best_epoch} epoch')
            logger.info(f'L1-norm      : {l1_norm:.3f}')
            logger.info(f'L2-norm      : {l2_norm:.3f}')
            logger.info(f'Linf-norm    : {inf_norm:.3f}')
            logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]:.3f}')
            writer.add_scalar("epoch/l1-norm", l1_norm, epoch)
            writer.add_scalar("epoch/l2-norm", l2_norm, epoch)
            writer.add_scalar("epoch/inf-norm", inf_norm, epoch)
            writer.add_scalar("epoch/learning rate", optimizer.param_groups[0]["lr"], epoch)
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            if beta_scheduler is not None:
                beta_scheduler.step(None)

    writer.flush()
    writer.close()
    
    uap_checkpoint = {
        'args': args,
        'epoch': epoch,
        'fr_list': fr_list,
        'fr': best_fr,
        'time': training_time,
        'uap': best_uap,
    }
    torch.save(uap_checkpoint, uap_filename)
    
    _print(f'The training took {training_time:,} seconds.')
    _print('Training completed.')
    _print(f'The best fooling rate was {best_fr:.3f}% and achived at the {best_epoch}-th epoch.')


if __name__ == '__main__':
    try:
        evaluation()
        batch_loss()
    except Exception as e:
        _print(f'Raised {e!r}')
