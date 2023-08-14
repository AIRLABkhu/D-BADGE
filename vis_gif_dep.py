from pathlib import Path
from glob import glob
from tqdm import tqdm
from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 13

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.transforms import ToTensor

import models, data

DEVICE = 'cuda:1'
boundary_targets = [1, 3, 4, 6]
target_indices = torch.tensor(boundary_targets).long()

model_dir = Path('./log/cifar10_resnet18_1.0/')
exp_dir = model_dir.joinpath('exp_until_80')
boundary_filename = model_dir.joinpath('decision.pth')

capture_filenames = sorted(glob(str('./temp/simba_result/*')))

model_ckpt = torch.load(model_dir.joinpath('ckpt.pth'))
net = models.get_model(model_ckpt['model']).to(DEVICE).eval()
net.load_state_dict(model_ckpt['net'])

evalset = data.get_dataset('cifar10', train=False, transform=ToTensor())
# sample_indices = [-1 for _ in range(10)]
# for i, (_, target) in enumerate(evalset):  # Find the first instances
#     if sample_indices[target] == -1:
#         sample_indices[target] = i
#     if -1 not in sample_indices:
#         break
# sample_indices = torch.tensor(sample_indices).long()[target_indices]
# indices = sample_indices
# evalset = Subset(evalset, indices)

indices = torch.tensor([i for i, (_, tar) in enumerate(tqdm(evalset, desc='FILTER_CLS')) if tar in boundary_targets]).long()
evalset = Subset(evalset, indices)

# ======================================================================================= #
@torch.no_grad()
def predict(pert=None, require_targets=False, require_feats=False):
    if pert is None:
        pert = torch.zeros(1, 3, 32, 32)
    loader = DataLoader(evalset, batch_size=512, num_workers=8)
        
    accm_preds, accm_targets, accm_feats = [], [], []
    for inputs, targets in tqdm(loader, desc='EVAL', leave=False, position=2):
        inputs = inputs.to(DEVICE) + pert.to(DEVICE)
        outputs, feats = net(inputs, acquire_feat=True)
        preds = outputs.argmax(dim=1)
        
        accm_preds.append(preds.cpu())
        if require_targets:
            accm_targets.append(targets)
        if require_feats:
            accm_feats.append(feats)
            
    result = [torch.cat(accm_preds, dim=0)]
    if require_targets:
        result.append(torch.cat(accm_targets, dim=0))
    if require_feats:
        result.append(torch.cat(accm_feats, dim=0))
    return result
# ======================================================================================= #
    
# Make decision boundary
boundary_ckpt = torch.load(boundary_filename)
size = boundary_ckpt['size']
boundary_targets = boundary_ckpt['boundary_targets']
principal_indices = boundary_ckpt['principal_indices']
decision_mesh_x, decision_mesh_y, decision_mesh_z = boundary_ckpt['decision_mesh']

# Load perturbations
# perts = [torch.cat(torch.load(fn), dim=0) for i, fn in enumerate(capture_filenames) if i in boundary_targets]
perts = torch.cat([torch.load(fn)[-1] for fn in capture_filenames], dim=0)
        
# preds_cln, targets, base_feats = predict(require_targets=True, require_feats=True)
# base_feats = base_feats.cpu()

# # Get features
# accm_feats = []
# with torch.no_grad():
#     for (image_adv, target), pert in zip(tqdm(evalset, desc='FORWARD'), perts):
#         batch_size = pert.size(0)
#         image_adv = image_adv.unsqueeze(0).expand_as(pert) + pert
        
#         loader = DataLoader(TensorDataset(image_adv), batch_size=512)
#         feats_list = []
#         for (image_adv,) in loader:
#             _, feats = net(image_adv.to(DEVICE), acquire_feat=True)
#             feats_list.append(feats)
#         accm_feats.append(torch.cat(feats_list, dim=0))
# feats = accm_feats

def render_frame(feat, sample_image, pert, num_update, total_num_updates):
    figsize = 2.0
    alpha = 1
    scatter_size = 16
    cmap = 'viridis'
    categories = data.get_categories('cifar10')
    target_categories = [categories[t] for t in boundary_targets]

    plt.figure(figsize=torch.tensor((6.0, 3.0)) * figsize)
    
    ax = plt.subplot(121)
    plt.title(f'{num_update}/{total_num_updates} updates')
    
    ax.contourf(decision_mesh_x, decision_mesh_y, decision_mesh_z, alpha=0.3, cmap=cmap)
    scatter = ax.scatter(feat[:, principal_indices[0]],
                         feat[:, principal_indices[1]],
                         c=boundary_targets, 
                         s=scatter_size, alpha=alpha, cmap=cmap)
    ax.legend(scatter.legend_elements()[0], target_categories, title='Categories')
    
    def imshow(ax, img):
        img_ = torch.clamp(img, 0, 1)
        ax.imshow(img_.permute(1, 2, 0))
    
    ax = plt.subplot(143)
    pert_ = (pert - pert.min()) / (pert.max() - pert.min())
    imshow(ax, pert_)
    
    ax = plt.subplot(144)
    imshow(ax, sample_image + pert)
    
    buffer = BytesIO()
    plt.savefig(buffer)
    img = Image.open(buffer)
    plt.close()
    return img

# num_frames = 400
# counts = torch.tensor([p.size(0) for p in perts])
# sub_counts = torch.ceil(counts / counts.sum() * num_frames).long().tolist()
# offsets = torch.cat([torch.tensor([0]), counts.cumsum(dim=0)], dim=0)

# frames = []
# num_update = 0
# import os
# os.system('mkdir temp/temp/')
# for i, (feat, (sample_image, target), pert) in enumerate(zip(feats, evalset, tqdm(perts, 'RENDER', leave=False, position=1))):
#     indices = torch.linspace(0, feat.size(0) - 1, sub_counts[i]).round().long()
#     feat, pert = feat[indices], pert[indices]
    
#     for j, (f, p) in enumerate(zip(feat, tqdm(pert, leave=False, position=2))):
#         base_feats[i] = f
#         img: Image.Image = render_frame(base_feats, sample_image, p, offsets[i] + indices[j], total_num_updates)
#         frames.append(img)
#         img.save(f'temp/temp/{len(frames):03d}.png')

# filename = 'simba'
# frames[0].save(f'temp/{filename}_first.png')
# frames[-1].save(f'temp/{filename}_last.png')
# frames[0].save(f'temp/{filename}.gif', save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)

indices = torch.load('temp/indices.pt')
evalset = Subset(evalset, indices)
eval_loader = DataLoader(evalset, batch_size=len(evalset), shuffle=False, num_workers=8)

inputs_cln, targets = next(iter(eval_loader))
inputs_cln = inputs_cln.to(DEVICE)
inputs_adv = inputs_cln + perts.to(DEVICE)

_, feats_cln = net(inputs_cln, acquire_feat=True)
_, feats_adv = net(inputs_adv, acquire_feat=True)

feats_cln = feats_cln.cpu() 
feats_adv = feats_adv.cpu() 

figsize = 2.0
alpha = 1
scatter_size = 16
cmap = 'viridis'
categories = data.get_categories('cifar10')
target_categories = [categories[t] for t in boundary_targets]
plt.figure(figsize=torch.tensor((6.0, 3.0)) * figsize)

ax = plt.subplot(121)
plt.title(f'After 2 hours 25 minutes')
ax.contourf(decision_mesh_x, decision_mesh_y, decision_mesh_z, alpha=0.3, cmap=cmap)
scatter = ax.scatter(feats_adv[:, principal_indices[0]],
                     feats_adv[:, principal_indices[1]],
                     c=targets, s=scatter_size, alpha=0.5, cmap=cmap)
ax.legend(scatter.legend_elements()[0], target_categories, title='Categories')

ax = plt.subplot(122)
plt.title(f'Original')
ax.contourf(decision_mesh_x, decision_mesh_y, decision_mesh_z, alpha=0.3, cmap=cmap)
scatter = ax.scatter(feats_cln[:, principal_indices[0]],
                     feats_cln[:, principal_indices[1]],
                     c=targets, s=scatter_size, alpha=0.5, cmap=cmap)
ax.legend(scatter.legend_elements()[0], target_categories, title='Categories')

plt.savefig('temp/simba_final.png')
