import os
import argparse
from datetime import datetime
from omegaconf import OmegaConf
from random import randint

__SEEDS__ = (952527575, 566433728, 115493150, 298539924, 731741520,
             443511375, 585143701, 455404733, 257820712, 439984570,
             298787371, 927585808, 822586280, 131408989, 166003373,
             125120996, 118786373, 104708221, 844471540, 729808726,)  # 20 seeds

OmegaConf.register_new_resolver('date', lambda: datetime.now().strftime('%y%m%d'))
OmegaConf.register_new_resolver('time', lambda: datetime.now().strftime('%H%M%S'))
OmegaConf.register_new_resolver('seed', lambda i: __SEEDS__[i] if i >= 0 else randint(1, 2 ** 32 - 1))
OmegaConf.register_new_resolver('eval', eval)


def load(*config, cfg_dir=None, args=None):
    root = './cfg'
    if cfg_dir is not None:
        root = os.path.join(root, cfg_dir)
        
    default_cfg = OmegaConf.load(os.path.join(root, 'default.yaml'))
    for c in config:
        if not c.endswith('.yaml'):
            c = c + '.yaml'
        cfg_filename = os.path.join(root, c)
        cfg = OmegaConf.load(cfg_filename)
        default_cfg.merge_with(cfg)
        
    if isinstance(args, argparse.Namespace):
        args = OmegaConf.create(vars(args))
    elif isinstance(args, dict):
        args = OmegaConf.create(args)
    elif isinstance(args, OmegaConf):
        pass
    else:
        raise TypeError('args must be dict, omegaconf.OmegaConf or argparse.Namespace.')
    
    for key in tuple(args.keys()):
        if args[key] is None:
            del args[key]
            
    default_cfg.merge_with(args)
    return default_cfg
        