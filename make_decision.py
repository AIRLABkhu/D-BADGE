import os
import matplotlib.pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 13
from tqdm import tqdm

from sklearn.svm import SVC

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import data, models

DEVICE = 'cuda:0'
dataset_name = 'cifar10'
resnet_depth = 18
exp_dir = f'./log/cifar10_resnet{resnet_depth}_1.0'
boundary_targets = [1, 3, 4, 6]

boundary_filename = os.path.join(exp_dir, 'decision.pth')

dataset = data.get_dataset(dataset_name, train=False, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=False, num_workers=12)
categories = data.get_categories(dataset_name)

model = models.get_model(f'resnet{resnet_depth}').eval().to(DEVICE)
model_checkpoint = torch.load(os.path.join(exp_dir, 'ckpt.pth'))
state_dict = model_checkpoint['net']
model.load_state_dict(state_dict)

uap = torch.load(os.path.join(exp_dir, '_baselines/00/ckpt.pth'))['uap'].to(DEVICE)

accm_onehots, accm_outputs_cln, accm_outputs_adv = [], [], []
accm_last_feats_cln, accm_last_feats_adv = [], []
with torch.no_grad():
    # Ground truth, logit, feature extraction
    for inputs_cln, targets in tqdm(loader, leave=False):
        batch_size = inputs_cln.size(0)
        onehots = F.one_hot(targets, len(data.get_categories('cifar10')))
        
        inputs_cln = inputs_cln.to(DEVICE)
        inputs_adv = inputs_cln.to(DEVICE) + uap
        inputs_all = torch.cat((inputs_cln, inputs_adv), dim=0)
        
        outputs, last_feats = model(inputs_all, acquire_feat=True)
        outputs_cln, outputs_adv = torch.split(outputs, batch_size)
        feats_cln, feats_adv = torch.split(last_feats, batch_size)
        
        accm_onehots.append(onehots)
        accm_outputs_cln.append(outputs_cln)
        accm_outputs_adv.append(outputs_adv)
        accm_last_feats_cln.append(feats_cln)
        accm_last_feats_adv.append(feats_adv)
        
    onehots = torch.cat(accm_onehots, dim=0).cpu()
    outputs_cln = torch.cat(accm_outputs_cln, dim=0).cpu()
    outputs_adv = torch.cat(accm_outputs_adv, dim=0).cpu()
    feats_cln = torch.cat(accm_last_feats_cln, dim=0).cpu()
    feats_adv = torch.cat(accm_last_feats_adv, dim=0).cpu()
    
    preds_cln = outputs_cln.argmax(dim=1)
    preds_adv = outputs_adv.argmax(dim=1)
    
    if not os.path.exists(boundary_filename):
        print('Finding principal components...')
        # PCA
        mean_cln = feats_cln.mean(dim=0, keepdim=True)
        centered_cln = feats_cln - mean_cln
        cov_mat = torch.matmul(centered_cln.T, centered_cln)
        eig_vals, eig_vecs = torch.linalg.eig(cov_mat.to(DEVICE))
        eig_vals = torch.real(eig_vals).cpu()
        eig_vecs = torch.real(eig_vecs).cpu()
        
        principal_indices = torch.argsort(eig_vecs[:, 0], descending=True)[:2]  # Top-2
        principal_feats_cln = feats_cln[:, principal_indices]
        principal_feats_adv = feats_adv[:, principal_indices]
        
        # Analyze principal components
        accm_target_match_map_cln = []
        accm_target_match_cln_indices = []
        accm_targets = []
        for i, target in enumerate(boundary_targets):
            target_match_map_cln = (outputs_cln.argmax(dim=1) == target)
            target_match_cln_indices = target_match_map_cln.nonzero().squeeze()
            
            accm_target_match_map_cln.append(target_match_map_cln)
            accm_target_match_cln_indices.append(target_match_cln_indices)
            accm_targets.append(torch.ones(target_match_cln_indices.size(0)) * i)
        target_match_map_cln = torch.cat(accm_target_match_map_cln, dim=0)
        target_match_cln_indices = torch.cat(accm_target_match_cln_indices, dim=0)
        targets = torch.cat(accm_targets, dim=0)
        target_categories = [categories[t] for t in boundary_targets]
        
        domain_padding = 0.03
        domain = feats_cln[target_match_cln_indices][:, principal_indices]
        domain_min = domain.min(dim=0).values
        domain_max = domain.max(dim=0).values
        domain_range = domain_max - domain_min
        domain_min -= domain_range * domain_padding
        domain_max += domain_range * domain_padding
        
        print('Finding decision boundary...')
        svc = SVC(C=len(boundary_targets))
        svc.fit(domain, targets)
        grid_size = 512
        decision_mesh_x, decision_mesh_y = torch.meshgrid(torch.linspace(domain_min[0], domain_max[0], grid_size),
                                                        torch.linspace(domain_min[1], domain_max[1], grid_size),
                                                        indexing='xy')
        decision_mesh_z = svc.predict(torch.hstack([decision_mesh_x.reshape(-1, 1), 
                                                    decision_mesh_y.reshape(-1, 1)])
                                    ).reshape(grid_size, grid_size)
        size = eig_vals.size(0)
        
        print('Saving a checkpoint...')
        torch.save(dict(
            size=size,
            boundary_targets=boundary_targets,
            principal_indices=principal_indices,
            decision_mesh=(decision_mesh_x, decision_mesh_y, decision_mesh_z),
        ), boundary_filename)
    else:
        print('Loading a checkpoint...')
        boundary_ckpt = torch.load(boundary_filename)
        size = boundary_ckpt['size']
        boundary_targets = boundary_ckpt['boundary_targets']
        principal_indices = boundary_ckpt['principal_indices']
        decision_mesh_x, decision_mesh_y, decision_mesh_z = boundary_ckpt['decision_mesh']
    
    alpha = 0.5
    scatter_size = 4
    title_y = -0.28
    cmap = 'viridis'
    plt.contourf(decision_mesh_x, decision_mesh_y, decision_mesh_z, alpha=0.3, cmap=cmap)
    plt.savefig('temp/temp.png')
    
    