import torch
from torch.utils.data import Dataset, Sampler


class ClassUniformSampler(Sampler):
    def __init__(self, data_source: Dataset, shuffle: bool=True):
        self.data_source = data_source
        self.shuffle = shuffle
        
        label_indices = {}
        for i, (_, label) in enumerate(self.data_source):
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(i)
        self.label_indices = [(label, torch.tensor(indices).long()) for label, indices in label_indices.items()]
        
    def __iter__(self):
        if self.shuffle:
            label_indices = [(label, indices[torch.randperm(len(indices))].tolist()) for label, indices in self.label_indices]
        else:
            label_indices = [(label, indices.tolist()) for label, indices in self.label_indices]
            
        while len(label_indices):
            to_del = []
            for i, (_, indices) in enumerate(label_indices):
                yield indices[0]
                
                if len(indices) == 1:
                    to_del.append(i)
                else:
                    indices.pop(0)
            
            for i in sorted(to_del, reverse=True):
                label_indices.pop(i)

    def __len__(self):
        return len(self.data_source)
    

from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torchvision.datasets import CIFAR10
                
dataset = CIFAR10('./data', train=True)
# sampler = ClassUniformSampler(dataset)
sampler = RandomSampler(dataset)
batch_sampler = BatchSampler(sampler, batch_size=4096, drop_last=False)
loader = DataLoader(dataset, batch_sampler=batch_sampler)

for inputs, targets in loader:
    print(targets)
    exit()
