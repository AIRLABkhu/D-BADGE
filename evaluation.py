'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import models, data 


def test(checkpoint: str, pert_checkpoint: str, tag: str, 
         device: str, visualize: bool=False, shuffle: bool=False, budget: float=-1):
    if pert_checkpoint is None:
        pert_checkpoint = checkpoint
    
    args = argparse.Namespace(
        checkpoint=checkpoint,
        pert_checkpoint=pert_checkpoint,
        tag=tag,
        device=device,
        visualize=visualize,
        shuffle=shuffle,
        budget=budget,
    )
    
    # Paths
    ckpt_dir = Path('./log').joinpath(args.checkpoint)
    ckpt_filename = ckpt_dir.joinpath('ckpt.pth')
    exp_dir = ckpt_dir.joinpath(args.tag)
    vis_dir = exp_dir.joinpath('visualization')
    log_filename = exp_dir.joinpath('eval.log')
    vis_dir.mkdir(parents=False, exist_ok=True)
    pert_dist_filename = vis_dir.joinpath('perturbation_distribution.png')
    pred_dist_filename = vis_dir.joinpath('prediction_distribution.png')
    pert_exp_dir = Path('./log', args.pert_checkpoint, args.tag)
    uap_filename = pert_exp_dir.joinpath('ckpt.pth')

    # Logging
    logger = logging.getLogger('EVAL')
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

    _print(args)

    # Device & Checkpoints
    print('==> LOADING CHECKPOINTS')
    DEVICE = args.device
    if DEVICE.lower().startswith('cuda'):
        torch.cuda.set_device(DEVICE)

    checkpoint = torch.load(ckpt_filename)
    uap_checkpoint = torch.load(uap_filename)
    _print('[Arguments]')
    if 'args' in checkpoint:
        _print('Victim:', checkpoint['args'])
    else:
        _print('Victim: unrecorded')
    if 'args' in uap_checkpoint:
        _print('UAP:', uap_checkpoint['args'])
    else:
        _print('UAP: unrecorded')

    # Dataset
    print('==> PREPAIRING DATA')
    testset = data.get_dataset(checkpoint['args'].dataset, train=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    categories = data.get_categories(checkpoint['args'].dataset)
    num_categories = len(categories)

    # Model
    print('==> BUILDING MODEL')
    model = models.get_model(checkpoint['model']).eval().to(DEVICE)
    model.load_state_dict(checkpoint['net'])
    uap = uap_checkpoint['uap'].to(DEVICE)
    
    if args.shuffle:
        pert_shape = uap.shape
        uap = uap.flatten(start_dim=2)
        pixel_indices = torch.randperm(n=uap.size(2)).long()
        uap = uap[:, :, pixel_indices]
        uap = uap.view(*pert_shape)
    if args.budget != -1:
        budget = args.budget / 255.0
        uap = uap.clamp(-budget, budget)

    with torch.no_grad():
        total_loss = 0
        correct_c, correct_u = 0, 0
        fooled, total = 0, 0
        confusion_mat_gt2clean = torch.zeros(num_categories, num_categories)
        confusion_mat_gt2uap = torch.zeros(num_categories, num_categories)
        confusion_mat_clean2uap = torch.zeros(num_categories, num_categories)
        with tqdm(testloader, desc='EVAL') as progress_bar:
            for inputs, targets in progress_bar:
                batch_size = inputs.size(0)
                
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                inputs_adv = torch.add(inputs, uap)
                
                inputs_all = torch.cat([inputs, inputs_adv], dim=0)
                inputs_all = torch.clamp(inputs_all, 
                                         min=inputs.amin(dim=(0, 2, 3), keepdim=True), 
                                         max=inputs.amax(dim=(0, 2, 3), keepdim=True))
                outputs_all = model(inputs_all)
                outputs_c, outputs_u = outputs_all.split(split_size=batch_size, dim=0)
                
                outputs_c = model(inputs)
                outputs_u = model(inputs_adv)
                
                predicted_c = outputs_c.argmax(dim=1).float()
                predicted_u = outputs_u.argmax(dim=1).float()
                
                targets = targets.float()
                total_loss = F.cross_entropy(predicted_c, targets, reduction='sum')
                
                total += inputs.size(0)
                correct_c += (predicted_c == targets).sum().item()
                correct_u += (predicted_u == targets).sum().item()
                fooled += (predicted_c != predicted_u).sum().item()
                
                for gt, class_c, class_u in zip(targets.cpu().long(), predicted_c.cpu().long(), predicted_u.cpu().long()):
                    confusion_mat_gt2clean[gt, class_c] += 1
                    confusion_mat_gt2uap[gt, class_u] += 1
                    confusion_mat_clean2uap[class_c, class_u] += 1

                mean_loss = total_loss / total
                accuracy_c = correct_c / total * 100
                accuracy_u = correct_u / total * 100
                fooling_rate = fooled / total * 100
                postfix_str = f'loss: {mean_loss:.3f} | v_acc: {accuracy_c:.3f} | fr: {fooling_rate:.3f}'
                progress_bar.set_postfix_str(postfix_str)
            
            _print( '[Victim]')
            _print(f'Cross entropy: {mean_loss:.4f}')
            _print(f'Accuracy:      {accuracy_c:.4f}%')
            _print( '[UAP]')
            _print(f'Accuracy:      {accuracy_u:.4f}%')
            _print(f'Fooling rate:  {fooling_rate:.4f}%')
            _print(f'L1-norm:       {uap.norm(p=1):.4f}')
            _print(f'Inf-norm:      {uap.norm(p=np.inf):.4f}')
            
        if args.visualize:
            uap = uap[0].cpu()
            
            print('==> VISUALIZATION')
            # Perturbation distribution
            print('Plotting perturbation distribution...')
            channels = [uap[0], uap[1], uap[2]]
            labels = ['r', 'g', 'b']
            colors = ['#F66', '#6F6', '#66F']
            plt.subplot(2, 2, 1)
            plt.title('perturbation')
            plt.imshow(uap.permute(1, 2, 0))

            plt.subplot(2, 2, 2)
            plt.title('grayscale')
            plt.hist(uap.mean(dim=0).view(-1), color='#666')

            for i, (channel, label, color) in enumerate(zip(channels, labels, colors), start=4):
                plt.subplot(2, 3, i)
                plt.title(label)
                plt.hist(channel.view(-1), color=color)
            plt.savefig(pert_dist_filename)
            
            if num_categories == 10:
                print('Plotting confusion matrices...')
                mats = (confusion_mat_gt2clean, confusion_mat_gt2uap, confusion_mat_clean2uap)
                titles = ('GT-Clean', 'UAP-GT', 'UAP-Clean')
                cmaps = ('YlGn', 'GnBu', 'OrRd')
                labels = (('GT', 'Clean'), ('UAP', 'GT'), ('UAP', 'Clean'))

                size = torch.tensor([len(mats), 1]) * 6
                plt.figure(figsize=size)
                for i, (mat, (x_label, y_label), title, cmap) in enumerate(zip(mats, labels, titles, cmaps), start=1):
                    ax = plt.subplot(1, len(mats), i)
                    
                    plt.title(title)
                    ax.matshow(mat, cmap=cmap)
                    for y in range(mat.size(0)):
                        for x in range(mat.size(1)):
                            ax.text(x, y, '%d' % mat[y, x], va='center', ha='center')
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.xaxis.set_label_position('top')
                    ax.set_xticks(range(num_categories))
                    ax.set_yticks(range(num_categories))
                    ax.set_xticklabels(categories, rotation=30, ha='left')
                    ax.set_yticklabels(categories, rotation=30)
            else:
                print('Plotting prediction histograms...')
                gt_count = confusion_mat_gt2clean.sum(dim=1)
                out_count = confusion_mat_gt2clean.sum(dim=0)
                uap_count = confusion_mat_clean2uap.sum(dim=0)
                
                counts = [gt_count, out_count, uap_count]
                labels = ['GT', 'Clean', 'UAP']
                colors = ['#3CC', '#C3C', '#CC3']

                uap_count_std = uap_count.float().std()
                peak_indices = torch.nonzero(uap_count > uap_count_std).squeeze()
                peak_out_count = out_count[peak_indices]
                peak_uap_count = uap_count[peak_indices]
                peaks = [None, peak_out_count, peak_uap_count]

                plt.figure(figsize=(10, 6))
                for i, (count, label, peak, color) in enumerate(zip(counts, labels, peaks, colors), start=1):
                    plt.subplot(len(counts), 1, i)
                    plt.title(label)
                    plt.bar(x=range(len(categories)), height=count, color=color)
                    if peak is not None:
                        plt.bar(x=peak_indices, height=peak, color='red')
            plt.savefig(pred_dist_filename)
            
    return accuracy_u, fooling_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 UAP Evaluation')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--pert-checkpoint', type=str, default=None)
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--budget', type=float, default=-1)
    parser.add_argument('--visualize', '-v', action='store_true', default=False)
    
    acc_u, fr = test(**vars(parser.parse_args()))
    neg_acc = 100.0 - acc_u
