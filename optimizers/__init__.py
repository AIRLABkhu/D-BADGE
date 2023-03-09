from torch import optim

name_optimizer_map = {
    name.lower(): t for name, t in optim.__dict__.items()
    if isinstance(t, type) and issubclass(t, optim.Optimizer) and t != optim.Optimizer
}

def get_optimizer(name: str, params, lr: float, **kwargs):
    if isinstance(name, str):
        name = name.lower()
    return name_optimizer_map[name](params=params, lr=lr, **kwargs)
