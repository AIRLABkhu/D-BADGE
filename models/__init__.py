from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnet_cifar import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .dla_simple import *
from .dla import *
from .mnist import *
from .toy import *
import functools
from torchvision import models as tv_models

model_map = {
    'vgg11':            functools.partial(VGG, 'VGG11'),
    'vgg13':            functools.partial(VGG, 'VGG13'),
    'vgg16':            functools.partial(VGG, 'VGG16'),
    'vgg19':            functools.partial(VGG, 'VGG19'),
    'resnet18':         ResNet18,
    'resnet101':        ResNet101,
    'resnet20':         resnet20,
    'densenet121':      DenseNet121,
    'preact_resnet18':  PreActResNet18,    
    'googlenet':        GoogLeNet,
    'resnext29_2x64d':  ResNeXt29_2x64d,
    'mobilenet':        MobileNet,
    'mobilenet_v2':     MobileNetV2,
    'dpn92':            DPN92,
    'shufflenet_g2':    ShuffleNetG2,
    'senet18':          SENet18,
    'shufflenet_v2':    functools.partial(ShuffleNetV2, 1),
    'efficientnet_b0':  EfficientNetB0,
    'regnetx_200mf':    RegNetX_200MF,
    'simple_dla':       SimpleDLA,
    
    'mnist_vgg11':      functools.partial(VGG, 'VGG11', in_channels=1),
    
    'mnist_arc_a':      mnist_arc_a,
    'mnist_arc_b':      mnist_arc_b,
    'mnist_arc_c':      mnist_arc_c,
    'mnist_arc_d':      mnist_arc_d,
    'mnist_arc_e':      mnist_arc_e,
    
    'toy_mnist':        toy_mnist,
    'toy_mnist_t':      toy_mnist_t,
    'toy_cifar10':      toy_cifar10,
    'toy_cifar100':     toy_cifar100,
    
    'tv_vgg16':         tv_models.vgg16,
    'tv_vgg19':         tv_models.vgg19,
    'tv_resnet50':      tv_models.resnet50,
    'tv_resnet101':     tv_models.resnet101,
    'tv_resnet152':     tv_models.resnet152,
    'tv_inception_v3':  tv_models.inception_v3,
}

tv_weights_map = {
    'tv_vgg16':         tv_models.VGG16_Weights.IMAGENET1K_V1,
    'tv_vgg19':         tv_models.VGG19_Weights.IMAGENET1K_V1,
    'tv_resnet50':      tv_models.ResNet50_Weights.IMAGENET1K_V1,
    'tv_resnet101':     tv_models.ResNet101_Weights.IMAGENET1K_V1,
    'tv_resnet152':     tv_models.ResNet152_Weights.IMAGENET1K_V1,
    'tv_inception_v3':  tv_models.Inception_V3_Weights.IMAGENET1K_V1,
}

tv_input_size_map = { # (model_dim, crop_size)
    'tv_vgg16':         (256      , 224      ),
    'tv_vgg19':         (256      , 224      ),
    'tv_resnet50':      (256      , 224      ),
    'tv_resnet101':     (256      , 224      ),
    'tv_resnet152':     (256      , 224      ),
    'tv_inception_v3':  (342      , 299      ),
}


def get_model(name: str):
    if name not in model_map:
        raise NotImplementedError(f'{name} is not implemented. Supported models: {", ".join(model_map.keys())}.')
    
    return model_map[name]()

def setup_tv_model_checkpoint(model_name: str, device: str=None, if_not_exists: bool=True, verbose: bool=True):
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    tv_model_names = [n for n in model_map.keys() if n.startswith('tv_')]
    if model_name not in model_map:
        raise NotImplementedError(f'{model_name} is not implemented. Supported models: {", ".join(tv_model_names)}.')
    if not model_name.startswith('tv_'):
        raise NotImplementedError(f'{model_name} is not a valid torchvision model. Supported models: {", ".join(tv_model_names)}.')
    
    import os
    from pathlib import Path
    
    def v_print(*values, sep=' ', end='\n'):
        if verbose:
            print(*values, sep=sep, end=end)
    
    log_dir = Path(f'./log/imagenet1k_{model_name[3:]}')
    if log_dir.exists():
        if if_not_exists:
            v_print(f'Log directory already exists: "{log_dir}".')
            return
        else:
            os.system(f'rm -rf {log_dir}')
        
    from tqdm import tqdm
    from typing import OrderedDict
    from torch.nn.functional import one_hot, cross_entropy
    from torch.utils.data import DataLoader
    import data
    
    v_print('==> Creating log directory')
    v_print('Log directory:', log_dir, '\n')
    log_dir.mkdir(parents=True, exist_ok=False)
    
    v_print('==> Downloading model parameters')
    v_print('Architecture:', model_name[3:])
    weights = tv_weights_map[model_name]
    v_print('Weights:', weights, '\n')
    model = model_map[model_name](weights=weights).eval().to(device)
    
    v_print('==> Loading evaluation dataset')
    v_print('Dataset: ImageNet-1K')
    model_dim, crop_size = tv_input_size_map[model_name]
    v_print('Model dimension:', model_dim)
    v_print('Center crop:', crop_size, '\n')
    transform = weights.transforms()
    testset = data.get_dataset('imagenet1k', train=False, transform=transform)
    loader = DataLoader(testset, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    
    v_print('==> Starting evaluation')
    with torch.no_grad(), tqdm(loader, desc='EVAL', dynamic_ncols=True) as progress_bar:
        total_loss = 0
        correct, total = 0, 0
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets_onehot = one_hot(targets, num_classes=1000).float()
            
            outputs = model(inputs).cpu()
            predictions = outputs.argmax(dim=1)
            
            total_loss += cross_entropy(outputs, targets_onehot, reduction='sum')
            correct += (targets == predictions).sum().item()
            total += inputs.size(0)
            
            mean_loss = total_loss / total
            accuracy = correct / total * 100.
            
            postfix_str = f'avg. ce_loss: {mean_loss:.2f} | accuracy: {accuracy:.4f}%'
            progress_bar.set_postfix_str(postfix_str)
    v_print(f'The accuracy of {model_name} on ImageNet-1K is {accuracy:.6f}%.')
    
    v_print('==> Saving checkpoint file')
    ckpt_filename = log_dir.joinpath('ckpt.pth')
    v_print('Checkpoint file name:', ckpt_filename, '\n')
    ckpt = {
        'args': {},
        'model': model_name,
        'net': OrderedDict({key: val.cpu() for key, val in model.state_dict().items()}),
        'epoch': -1,
        'acc': accuracy,
    }
    torch.save(ckpt, ckpt_filename)
