from functools import partial
from torch import nn

def toy_mnist_t(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, 128), nn.ReLU(),
        nn.Linear(128, num_classes),
    )

def toy_mnist(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3, 3)) , nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=(3, 3)), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
        nn.Conv2d(32, 64, kernel_size=(3, 3)), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3)), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
        nn.Flatten(),
        nn.Linear(1024, 200), nn.ReLU(inplace=True),
        nn.Linear(200, 200) , nn.ReLU(inplace=True),
        nn.Linear(200, num_classes)
    )
    
def toy_cifar(num_classes: int=10):
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3))   , nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3))  , nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
        nn.Conv2d(64, 128, kernel_size=(3, 3)) , nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=(3, 3)), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
        nn.Flatten(),
        nn.Linear(3200, 256), nn.ReLU(inplace=True),
        nn.Linear(256, 256) , nn.ReLU(inplace=True),
        nn.Linear(256, num_classes)
    )

toy_cifar10  = partial(toy_cifar, 10)
toy_cifar100 = partial(toy_cifar, 100)


