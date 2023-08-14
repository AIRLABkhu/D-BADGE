''' 
CNNs for FashionMNIST dataset classification.
Implementation of ArcA, ArcB, ArcC, ArcD and ArcE from 
"Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations" CVPR 2022.

Detailed architecture wasn't written in the paper, thus we selected following components by ourselves:
    - the number of channels of convolution layers,
    - classification head.
'''

from functools import partial
from torch import nn


def mnist(activation, use_dropout, kernel_size, num_conv_layers, num_classes=10):
    layers = []
    for i in range(num_conv_layers):
        if i == 0:
            in_channels = 1
        else:
            in_channels = 64
        
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=kernel_size))
        layers.append(activation)
        if use_dropout:
            layers.append(nn.Dropout2d(p=0.2, inplace=False))
    layers.append(nn.AdaptiveAvgPool2d(output_size=1))
    layers.append(nn.Flatten())
    
    layers.append(nn.Linear(64, 32))
    layers.append(activation)
    if use_dropout:
        layers.append(nn.Dropout1d(p=0.2, inplace=False))
    layers.append(nn.Linear(32, num_classes))
    layers.append(nn.Sigmoid())
    
    return nn.Sequential(*layers)
        
import torch.autograd
torch.autograd.set_detect_anomaly(True)
__config = dict(a=dict(activation       =nn.ELU(inplace=True),
                       use_dropout      =False,
                       kernel_size      =3,
                       num_conv_layers  =2),
                b=dict(activation       =nn.ELU(inplace=True),
                       use_dropout      =False,
                       kernel_size      =5,
                       num_conv_layers  =3),
                c=dict(activation       =nn.ReLU(inplace=True),
                       use_dropout      =False,
                       kernel_size      =3,
                       num_conv_layers  =2),
                d=dict(activation       =nn.PReLU(),
                       use_dropout      =False,
                       kernel_size      =3,
                       num_conv_layers  =4),
                e=dict(activation       =nn.ReLU(inplace=False),
                       use_dropout      =True,
                       kernel_size      =5,
                       num_conv_layers  =2))

mnist_arc_a = partial(mnist, **__config['a'])
mnist_arc_b = partial(mnist, **__config['b'])
mnist_arc_c = partial(mnist, **__config['c'])
mnist_arc_d = partial(mnist, **__config['d'])
mnist_arc_e = partial(mnist, **__config['e'])
    
    
if __name__ == '__main__':
    import torch
    sample = torch.randn(32, 1, 28, 28)
    for name, opt in __config.items():
        net = mnist(**opt)
        net(sample)
    print('PASSED:', __file__)

            