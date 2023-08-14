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

model_dir = Path('./log/cifar10_resnet18_1.0/')
exp_dir = model_dir.joinpath('exp_until_80')
boundary_filename = model_dir.joinpath('decision.pth')

model_ckpt = torch.load(model_dir.joinpath('ckpt.pth'))
net = models.get_model(model_ckpt['model']).to(DEVICE).eval()
net.load_state_dict(model_ckpt['net'])

sample_indices = [1251, 1252, 1253]
full_evalset = data.get_dataset('cifar10', train=False, transform=ToTensor())
samples = [full_evalset[i] for i in sample_indices]
sample_images = [s[0] for s in samples]
sample_targets = [s[1] for s in samples]

indices = torch.tensor([i for i, (_, tar) in enumerate(tqdm(full_evalset, desc='FILTER_CLS')) if tar in boundary_targets]).long()
full_evalset = Subset(full_evalset, indices)
eval_loader = DataLoader(full_evalset, batch_size=512, shuffle=False, num_workers=8)

# ======================================================================================= #
@torch.no_grad()
def predict(pert=None, require_targets=False):
    if pert is None:
        pert = torch.zeros(1, 3, 32, 32)
    accm_preds, accm_targets = [], []
    for inputs, targets in tqdm(eval_loader, desc='EVAL', leave=False, position=2):
        inputs = inputs.to(DEVICE) + pert.to(DEVICE)
        outputs = net(inputs)
        preds = outputs.argmax(dim=1)
        
        accm_preds.append(preds.cpu())
        if require_targets:
            accm_targets.append(targets)
            
    if require_targets:
        return torch.cat(accm_preds, dim=0), torch.cat(accm_targets, dim=0)
    else:
        return torch.cat(accm_preds, dim=0)
# ======================================================================================= #

# Decision-BADGE #
capture_filenames = sorted(glob(str(exp_dir.joinpath('captures_*.pth'))))
captures = [torch.load(fn) for fn in capture_filenames]
uaps = []
for capt in captures:
    for uap, _, num_updates in capt:
        uaps.append(uap.cpu())

# UAP #
# capture_filenames = sorted(glob('./temp/uap_p/*'))
# uaps = sum([torch.load(fn) for fn in tqdm(capture_filenames, desc='CAPT')], [])

final_uap = uaps[-1]
total_num_updates = len(uaps)
        
preds_cln, targets = predict(require_targets=True)
torch.manual_seed(3829)
indices = torch.randperm(len(full_evalset))[40:]
# ok_map, indices_map = {}, {}
# with tqdm(total=int(ceil(log2(len(uaps)))), desc='SRCH', leave=False, position=1) as srch_bar:
#     left, right = 0, len(uaps) - 1
#     while left <= right:
#         mid = (right + left) // 2
#         if left in ok_map:
#             ok_left = ok_map[left]
#         else:
#             preds_left = predict(uaps[left])
#             indices_left = torch.logical_and(preds_cln == targets, preds_cln != preds_left).nonzero().flatten()
#             fooled_classes_left = set(targets[indices_left].tolist())
#             ok_left = fooled_classes_left == set(boundary_targets)
#             ok_map[left] = ok_left
#             indices_map[left] = indices_left
        
#         if mid in ok_map:
#             ok_mid = ok_map[mid]
#         else:
#             preds_mid = predict(uaps[mid])
#             indices_mid = torch.logical_and(preds_cln == targets, preds_cln != preds_mid).nonzero().flatten()
#             fooled_classes_mid = set(targets[indices_mid].tolist())
#             ok_mid = fooled_classes_mid == set(boundary_targets)
#             ok_map[mid] = ok_mid
#             indices_map[mid] = indices_mid
        
#         if ok_mid != ok_left:
#             right = mid - 1
#         else:
#             left = mid + 1
#         srch_bar.update()
#     print()
    
# Find the first decision
# indices = indices_map[left]
unique_count = {}
unique_indices = []
for i, t in zip(indices.tolist(), targets[indices].tolist()):
    if t not in unique_count:
        unique_count[t] = 0
        
    if unique_count[t] < 10:
        unique_count[t] += 1 
        unique_indices.append(i)
        
indices = torch.tensor(unique_indices).long()
indices = indices[targets[indices].argsort()]

n_frames = 200
last_frame = 45000
uaps = torch.cat(uaps, dim=0)
if n_frames == -1:
    n_frames = len(uaps)
else:
    num_updates = torch.linspace(0, last_frame, n_frames).round().long()
    # num_updates = torch.tensor([50000, 50001]).long()
    # n_frames = 2
    # last_frame = 50001
    uaps = uaps[num_updates]

filteredset = Subset(full_evalset, indices)
filtered_loader = DataLoader(filteredset, batch_size=len(indices), shuffle=False, num_workers=8)
filtered_cln, targets = next(iter(filtered_loader))
filtered_adv = filtered_cln[None].repeat(n_frames, 1, 1, 1, 1) + \
               uaps[:, None].repeat(1, len(indices), 1, 1, 1)
targets = targets[None].repeat(n_frames, 1)

# Make adversarial example dataset
filteredset = TensorDataset(filtered_adv.flatten(0, 1), targets.flatten(0, 1))
filtered_loader = DataLoader(filteredset, batch_size=512, shuffle=False, num_workers=8)

accm_feats = []
accm_targets = []
for inputs, targets in tqdm(filtered_loader, desc='FORWARD'):
    batch_size = inputs.size(0)
    inputs = inputs.to(DEVICE)
    
    _, last_feats = net(inputs, acquire_feat=True)
    accm_feats.append(last_feats)
    accm_targets.append(targets)
    
feats = torch.cat(accm_feats, dim=0).reshape(n_frames, len(indices), -1).cpu()
targets = torch.cat(accm_targets, dim=0).reshape(n_frames, len(indices)).cpu()
    
# Make decision boundary
boundary_ckpt = torch.load(boundary_filename)
size = boundary_ckpt['size']
boundary_targets = boundary_ckpt['boundary_targets']
principal_indices = boundary_ckpt['principal_indices']
decision_mesh_x, decision_mesh_y, decision_mesh_z = boundary_ckpt['decision_mesh']

figsize = 2.0
alpha = 1
scatter_size = 16
cmap = 'viridis'
categories = data.get_categories('cifar10')
target_categories = [categories[t] for t in boundary_targets]

frames = []
# for i, (feat, target, uap, num_update) in enumerate(zip(feats, targets, tqdm(uaps, desc='RENDER'), num_updates), start=1):
#     plt.figure(figsize=torch.tensor((6.0, 3.0)) * figsize)
    
#     ax = plt.subplot(121)
#     plt.title(f'{num_update}/{num_updates[-1]} updates')
    
#     ax.contourf(decision_mesh_x, decision_mesh_y, decision_mesh_z, alpha=0.3, cmap=cmap)
#     scatter = ax.scatter(feat[:, principal_indices[0]],
#                          feat[:, principal_indices[1]],
#                          c=target, s=scatter_size, alpha=alpha, cmap=cmap)
#     ax.legend(scatter.legend_elements()[0], target_categories, title='Categories')
    
#     def imshow(ax, img):
#         img_ = torch.clamp(img, 0, 1)
#         ax.imshow(img_.permute(1, 2, 0))
    
#     ax = plt.subplot(243)
#     uap_ = (uap - uap.min()) / (uap.max() - uap.min())
#     imshow(ax, uap_)
    
#     ax = plt.subplot(244)
#     imshow(ax, sample_images[0] + uap)
    
#     ax = plt.subplot(247)
#     imshow(ax, sample_images[1] + uap)
    
#     ax = plt.subplot(248)
#     imshow(ax, sample_images[2] + uap)
    
#     buffer = BytesIO()
#     plt.savefig(buffer)
#     img = Image.open(buffer)
#     plt.close()
#     frames.append(img)

filename = 'badge'
# frames[0].save(f'temp/{filename}_first.png')
# frames[-1].save(f'temp/{filename}_last.png')
# frames[0].save(f'temp/{filename}.gif', save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)

# Render the final frame
plt.figure(figsize=torch.tensor((6.0, 3.0)) * figsize)

with torch.no_grad():
    accm_targets = []
    accm_preds_cln, accm_preds_adv = [], []
    accm_feats_cln, accm_feats_adv = [], []
    rand_indices = torch.randperm(len(full_evalset))[3275:]
    torch.save(rand_indices, 'temp/indices.pt')
    subset = Subset(full_evalset, rand_indices)
    subset_loader = DataLoader(subset, batch_size=512, shuffle=False)
    for inputs_cln, targets in tqdm(subset_loader, desc='FULL_FWD'):
        inputs_cln = inputs_cln.to(DEVICE)
        inputs_adv = inputs_cln + final_uap.to(DEVICE)
        
        outputs_cln, feats_cln = net(inputs_cln, acquire_feat=True)
        outputs_adv, feats_adv = net(inputs_adv, acquire_feat=True)
        
        accm_targets.append(targets)
        accm_preds_cln.append(outputs_cln.argmax(dim=1))
        accm_preds_adv.append(outputs_adv.argmax(dim=1))
        accm_feats_cln.append(feats_cln)
        accm_feats_adv.append(feats_adv)
    targets = torch.cat(accm_targets, dim=0).cpu()
    preds_cln = torch.cat(accm_preds_cln, dim=0).cpu()
    preds_adv = torch.cat(accm_preds_adv, dim=0).cpu()
    feats_cln = torch.cat(accm_feats_cln, dim=0).cpu()
    feats_adv = torch.cat(accm_feats_adv, dim=0).cpu()

print('RENDERING THE FINAL FRAME...')
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

def imshow(ax, img):
    if img.ndim == 4:
        img = img[0]
    img_ = torch.clamp(img, 0, 1)
    ax.imshow(img_.permute(1, 2, 0))

plt.savefig(f'temp/{filename}_final_725.png')
