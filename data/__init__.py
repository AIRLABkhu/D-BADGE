import os

import torch
from torchvision import datasets, transforms

from data import augmentations


DATAROOT = os.path.dirname(__file__)
IMAGENET1K_ROOT = '/material/data/imagenet-original'

supported_datasets = (
    'mnist',
    'mnist_3ch',
    'cifar10',
    'cifar100',
    'imagenet1k',
)

class MNIST_3ch(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        resize_channel_expand = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(32, 32)),
            transforms.Lambda(lambda x: x.expand(3, 32, 32)),
            transforms.ToPILImage(),
        ])
        if transform is None:
            transform = resize_channel_expand
        else:
            transform = transforms.Compose([resize_channel_expand, transform])
        super(MNIST_3ch, self).__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)


def check_dataset_support(dataset: str):
    dataset = dataset.lower()
    if dataset not in supported_datasets:
        raise NotImplementedError(f'{dataset} is not supported. Please select one of ({", ".join(supported_datasets)}).')

def non_case_sensitive_str(data):
    return str(data).lower()

def get_dataset(dataset: str, train: bool, transform=None, target_transform=None):
    dataset = dataset.lower()
    check_dataset_support(dataset)
    
    if dataset == 'mnist':
        return datasets.MNIST(root=DATAROOT, train=train, transform=transform, target_transform=target_transform, download=True)
    elif dataset == 'mnist_3ch':
        return MNIST_3ch(root=DATAROOT, train=train, transform=transform, target_transform=target_transform, download=True)
    elif dataset == 'cifar10':
        return datasets.CIFAR10(root=DATAROOT, train=train, transform=transform, target_transform=target_transform, download=True)
    elif dataset == 'cifar100':
        return datasets.CIFAR100(root=DATAROOT, train=train, transform=transform, target_transform=target_transform, download=True)
    elif dataset == 'imagenet1k':
        root = os.path.join(IMAGENET1K_ROOT, 'train' if train else 'val')
        return datasets.ImageFolder(root=root, transform=transform, target_transform=target_transform)
    else:
        raise AssertionError('This code should not be run.')
    
def get_mean_std(dataset: str, as_tensor: bool=False):
    dataset = dataset.lower()
    check_dataset_support(dataset)
    
    if dataset == 'mnist':
        mean, std = (0.1307,), (0.3081,)
    elif dataset == 'mnist_3ch':
        mean, std = (0.1307, 0.1307, 0.1307), (0.3081, 0.3081 ,0.3081)
    elif dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    elif dataset == 'imagenet1k':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        raise AssertionError('This code should not be run.')
    
    if as_tensor:
        return torch.tensor(mean).view(1, 1, -1), torch.tensor(std).view(1, 1, -1)
    else:
        return mean, std

def get_categories(dataset: str):
    dataset = dataset.lower()
    check_dataset_support(dataset)
    
    if dataset == 'mnist':
        cats = [str(i) for i in range(10)]
    elif dataset == 'mnist_3ch':
        cats = [str(i) for i in range(10)]
    elif dataset == 'cifar10':
        cats = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == 'cifar100':
        cats = sorted(['beaver', 'dolphin', 'otter', 'seal', 'whale',  # ......................| aquatic mammals
                       'aquarium' 'fish', 'flatfish', 'ray', 'shark', 'trout',  # .............| fish
                       'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', # ...............| flowers
                       'bottles', 'bowls', 'cans', 'cups', 'plates', # ........................| food containers
                       'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', # ..........| fruit and vegetables
                       'clock', 'computer' 'keyboard', 'lamp', 'telephone', 'television', # ...| household electrical devices
                       'bed', 'chair', 'couch', 'table', 'wardrobe', # ........................| household furniture
                       'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', # ............| insects
                       'bear', 'leopard', 'lion', 'tiger', 'wolf', # ..........................| large carnivores
                       'bridge', 'castle', 'house', 'road', 'skyscraper', # ...................| large man-made outdoor things
                       'cloud', 'forest', 'mountain', 'plain', 'sea', # .......................| large natural outdoor scenes
                       'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', # .............| large omnivores and herbivores
                       'fox', 'porcupine', 'possum', 'raccoon', 'skunk', # ....................| medium-sized mammals
                       'crab', 'lobster', 'snail', 'spider', 'worm', # ........................| non-insect invertebrates
                       'baby', 'boy', 'girl', 'man', 'woman', # ...............................| people
                       'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', # ................| reptiles
                       'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', # ...................| small mammals
                       'maple', 'oak', 'palm', 'pine', 'willow', # ............................| trees
                       'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', # .............| vehicles 1
                       'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor' # ...............| vehicles 2
        ])
    elif dataset == 'imagenet1k':
        with open(os.path.join(DATAROOT, 'imagenet1k_categories.txt'), 'r') as file:
            cats = eval(file.readline())
    else:
        raise AssertionError('This code should not be run.')
    
    return cats

def get_num_input_channels(dataset: str):
    dataset = dataset.lower()
    check_dataset_support(dataset)
    
    if dataset in {'mnist'}:
        return 1
    elif dataset in {'mnist_3ch', 'cifar10', 'cifar100', 'imagenet1k'}:
        return 3
    else:
        raise AssertionError('This code should not be run.')
