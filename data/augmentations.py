import os
from functools import partial
from omegaconf import OmegaConf
import torch
import numpy as np
from torchvision import transforms
from torchvision import models as tv_models
from PIL import Image


class RandomNoise:
    def __init__(self, mean, std):
        if isinstance(mean, (int, float)):
            mean = (mean,)
        if isinstance(std, (int, float)):
            std = (std,)
        
        self.mean = np.array(mean)
        self.std = np.array(std)
        
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            img = x
        elif isinstance(x, Image.Image):
            img = np.array(x, dtype=np.uint8).astype(np.float32) / 255.0
        elif isinstance(x, torch.Tensor):
            img = x.permute(1, 2, 0).numpy()
        else:
            raise TypeError(f'\'x\' (position 1) must be ndarray, Image or Tensor, not {type(x)}.')
        
        mean = self.mean.reshape(1, 1, -1)
        std = self.std.reshape(1, 1, -1)
        img = (img + np.random.randn(*img.shape)) * std + mean
        
        if isinstance(x, np.ndarray):
            return img
        elif isinstance(x, Image.Image):
            img = img * 255.0
            img = img.astype(np.uint8)
            return Image.fromarray(img)
        elif isinstance(x, torch.Tensor):
            return torch.from_numpy(img).permute(2, 0, 1)


augmentations = {
    'resize':           transforms.Resize,
    'center_crop':      transforms.CenterCrop,
    'random_crop':      transforms.RandomCrop,
    'h_flip':           transforms.RandomHorizontalFlip,
    'v_flip':           transforms.RandomVerticalFlip,
    'auto_cifar10':     partial(transforms.AutoAugment, transforms.AutoAugmentPolicy.CIFAR10),
    'auto_imagenet':    partial(transforms.AutoAugment, transforms.AutoAugmentPolicy.IMAGENET),
    
    'random_noise':     RandomNoise,
    
    'tv_vgg16':         tv_models.VGG16_Weights.IMAGENET1K_V1.transforms,
    'tv_vgg19':         tv_models.VGG19_Weights.IMAGENET1K_V1.transforms,
    'tv_resnet50':      tv_models.ResNet50_Weights.IMAGENET1K_V1.transforms,
    'tv_resnet101':     tv_models.ResNet101_Weights.IMAGENET1K_V1.transforms,
    'tv_resnet152':     tv_models.ResNet152_Weights.IMAGENET1K_V1.transforms,
    'tv_inception_v3':  tv_models.Inception_V3_Weights.IMAGENET1K_V1.transforms,
}

def register_augmentation(key, augmentation):
    if key in augmentations:
        raise KeyError(f'This key has value: {augmentation!r}.')

    augmentations[key] = augmentation

def build_transform(*names):
    filenames = []
    
    base_cfg = OmegaConf.create()
    for name in names:
        if not name.endswith('.yaml'):
            name += '.yaml'
        filenames.append(name)
        
        cfg = OmegaConf.load(os.path.join('./cfg/aug/', name))
        base_cfg.merge_with(cfg)

    transform = []
    for key, kwargs in base_cfg.items():
        if key in augmentations:
            if kwargs is None:
                kwargs = {}
            transform.append(augmentations[key](**kwargs))
        else:
            raise KeyError(f'Unknown augmentation: "{key}".')
        
    return transform, filenames


if __name__ == '__main__':
    from PIL import Image
    img = np.random.randint(255, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    
    trans = RandomNoise(mean=0, std=1)
    trans(img)
