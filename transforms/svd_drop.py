import torch


class SvdDrop:
    def __init__(self, major_drop: float=0, minor_drop: float=0.3, device: str=None):
        if major_drop + minor_drop >= 1:
            raise RuntimeError('The sum of drop rate must be in inclusive 0 and exclusive 1.')
        self.major_drop = major_drop
        self.minor_drop = minor_drop
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
    
    def __call__(self, x: torch.Tensor):
        x_shape = x.shape
        if x.ndim == 4:
            x = x.flatten(start_dim=0, end_dim=1)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise RuntimeError('Must provide an image tensor with shape (H, W) or (C, H, W).')
        x = x.to(self.device)
        
        u, s, v = torch.svd(x)
        major_keep_map = SvdDrop.__get_keep_map(s, threshold=self.major_drop, reverse=False)
        minor_keep_map = SvdDrop.__get_keep_map(s, threshold=self.minor_drop, reverse=True)
        keep_map_1d = torch.logical_and(major_keep_map, minor_keep_map)
        
        s = s * keep_map_1d
        s = torch.diag_embed(s)
            
        x = torch.bmm(u, torch.bmm(s, v.permute(0, 2, 1)))
        return x.view(*x_shape).cpu()
        
    
    @staticmethod
    def __get_keep_map(values: torch.Tensor, threshold: float, reverse: bool=False):
        if reverse:
            values = values.fliplr()  # reverse tensor
        
        v_cumsum = values.cumsum(dim=1)
        v_sum = v_cumsum[:, -1:]
        threshold = v_sum * threshold
        safe_map = (v_cumsum > threshold)  # to keep: 1, to drop: 0
        
        if reverse:
            safe_map = safe_map.fliplr()
        return safe_map
