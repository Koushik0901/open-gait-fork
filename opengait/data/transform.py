from data import transform as base_transform
import numpy as np
import torch
import torchvision.transforms as T
from utils import is_list, is_dict, get_valid_args


class NoOperation():
    def __call__(self, x):
        return x


class BaseSilTransform():
    def __init__(self, divsor=255.0, img_shape=None):
        self.divsor = divsor
        self.img_shape = img_shape

    def __call__(self, x):
        if self.img_shape is not None:
            s = x.shape[0]
            _ = [s] + [*self.img_shape]
            x = x.reshape(*_)
        return x / self.divsor


class BaseSilCuttingTransform():
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        x = x[..., cutting:-cutting]
        return x / self.divsor


class BaseRgbTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485*255, 0.456*255, 0.406*255]
        if std is None:
            std = [0.229*255, 0.224*255, 0.225*255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):
        return (x - self.mean) / self.std

class RandomHorizontalFlip:
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()
        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [-1])
        return tensor

def get_transform(trf_cfg=None, training=True):
    if training:
        transform = T.Compose([
            BaseSilCuttingTransform(),
            T.Lambda(lambda x: x.unsqueeze(1)),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.RandomAdjustSharpness(sharpness_factor=2, p=trf_cfg["p"]),
            T.Lambda(lambda x: x.squeeze()),
        ])
        return [transform]
    else:
        return [BaseSilCuttingTransform()]

# def get_transform(trf_cfg=None):
#     if is_dict(trf_cfg):
#         transform = getattr(base_transform, trf_cfg['type'])
#         valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
#         return transform(**valid_trf_arg)
#     if trf_cfg is None:
#         return lambda x: x
#     if is_list(trf_cfg):
#         transform = [get_transform(cfg) for cfg in trf_cfg]
#         return transform
#     raise "Error type for -Transform-Cfg-"
