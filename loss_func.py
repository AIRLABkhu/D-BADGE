import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=0.5, logits=False, reduce=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        nn.CrossEntropyLoss()

        eps = 1e-8
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)
        # F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        F_loss = self.alpha * (1 - pt + eps) ** self.gamma * (-torch.log(pt+eps))

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class AccuracyLoss(nn.Module):
    def __init__(self):
        super(AccuracyLoss, self).__init__()
        
    def forward(self, inputs, targets):
        scores = inputs + targets - 1
        zeros = torch.zeros_like(scores)
        return torch.max(zeros, scores).sum() / inputs.size(0)

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()
        
    def forward(self, inputs, targets):
        normalized_inputs = inputs / inputs.sum(dim=1, keepdim=True)
        normalized_targets = targets / inputs.sum(dim=1, keepdim=True)
        
        cdf_inputs = torch.cumsum(normalized_inputs, dim=1)
        cdf_targets = torch.cumsum(normalized_targets, dim=1)
        
        grad = torch.sum(torch.abs(cdf_targets - cdf_inputs)) / inputs.size(1)
        return grad
    

loss_functions = {
    'accuracy': AccuracyLoss(),
    'acc': AccuracyLoss(),
    'kld': lambda x, y: -nn.functional.kl_div(x, y, reduction='batchmean'),
    'ce': lambda x, y: -nn.functional.cross_entropy(x, y),
    'emd': EMDLoss(),
}

def get_loss_function(name: str):
    return loss_functions[name]
