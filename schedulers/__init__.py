import torch
from torch.optim import SGD, lr_scheduler


class Scheduler:
    def __init__(self, scheduler_type: type, value: float, **kwargs):
        self.scheduler_type = scheduler_type
        self.initial_value = value
        self.kwargs = kwargs
        
        dummy_param = [torch.tensor(0, dtype=torch.float, requires_grad=True)]
        self.__dummy_optim = SGD(dummy_param, lr=value)
        if self.scheduler_type is None:
            self.__scheduler = None
        else:
            self.__scheduler = scheduler_type(optimizer=self.__dummy_optim, **kwargs)
        
    def step(self, *args, **kwargs):
        if self.scheduler_type is not None:
            self.__dummy_optim.step()
            self.__scheduler.step(*args, **kwargs)
    
    @property
    def value(self):
        return self.__scheduler.get_last_lr()[0] if self.__scheduler else self.initial_value


name_scheduler_map = {
    name.lower(): t for name, t in lr_scheduler.__dict__.items()
    if isinstance(t, type) and issubclass(t, lr_scheduler._LRScheduler) and t != lr_scheduler._LRScheduler
}
name_scheduler_map[None] = None


def get_scheduler(name: str, value: float=None, optimizer=None, **kwargs):
    if isinstance(name, str):
        name = name.lower()

    if value is not None and optimizer is not None:
        raise ValueError('Must provide one of value or optimizer not both.')
    
    if value is not None:
        return Scheduler(name_scheduler_map[name], value=value, **kwargs)
    else:
        return name_scheduler_map[name](optimizer=optimizer, **kwargs)
