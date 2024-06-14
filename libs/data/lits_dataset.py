import os
import monai
from monai import transforms
from monai import data
import numpy as np
import torch
from monai.transforms import Transform, MapTransform
from monai.config import KeysCollection
from collections.abc import Callable, Hashable, Mapping, Sequence
from monai.utils import MAX_SEED, ensure_tuple, first


class LiTS(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, label="liver", allow_missing_keys: bool = False) -> None:
        self.label = label
        self.keys: tuple[Hashable, ...] = ensure_tuple(keys)
        self.allow_missing_keys = allow_missing_keys
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            if self.label == 'liver':
                result.append(torch.logical_or(d[key] == 1, d[key] == 2))
            elif self.label == 'tumor':
                result.append(d[key] == 2)
            else:
                result.append(d[key] == 1)
                result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


def lits_dataloader(datalist, batch_size, stage, shuffle, label):
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.EnsureTyped(keys=["image"], device=device),
                LiTS(keys='label', label=label),
                transforms.SpatialPadd(keys=["image","label"], spatial_size=(-1,-1,128)),
                transforms.Resized(keys=["image","label"], spatial_size=(256,256,-1)),
                transforms.RandCropByPosNegLabeld(keys=["image","label"], label_key="label",spatial_size=(128,128,128), pos=0.9, neg=0.1, num_samples=1, image_key='image'),
                
                transforms.ScaleIntensityRanged(keys=["image"],a_min=-200,a_max=250,b_min=0.0,b_max=1.0,clip=True,),
                transforms.ThresholdIntensityd(keys=["image"], threshold=0.35),
                transforms.MedianSmoothd(keys=["image"], radius=1),
            ]
        )
        
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.EnsureTyped(keys=["image"], device=device),
                LiTS(keys='label', label=label),
                transforms.SpatialPadd(keys=["image","label"], spatial_size=(-1,-1,128)),
                transforms.Resized(keys=["image","label"], spatial_size=(256,256,-1)),
                transforms.ScaleIntensityRanged(keys=["image"],a_min=-200,a_max=250,b_min=0.0,b_max=1.0,clip=True,),
                transforms.ThresholdIntensityd(keys=["image"], threshold=0.35),
                transforms.MedianSmoothd(keys=["image"], radius=1),
            ]
        )  

        test_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.EnsureTyped(keys=["image"], device=device),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                transforms.Resized(keys=["image"], spatial_size=(256,256,-1)),
                transforms.ScaleIntensityRanged(keys=["image"],a_min=-200,a_max=250,b_min=0.0,b_max=1.0,clip=True,),
                transforms.ThresholdIntensityd(keys=["image"], threshold=0.35),
                transforms.MedianSmoothd(keys=["image"], radius=1),
            ]
        )  
        
        if stage == 'train':
            train_set = data.Dataset(datalist['training'], transform= train_transform)
            valid_set = data.Dataset(datalist['validation'], transform= val_transform)
            
            train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
            valid_loader = data.DataLoader(valid_set, batch_size=1, shuffle=False)

            return train_loader, valid_loader
        
        elif stage == 'valid':
            valid_set = data.Dataset(datalist['validation'], transform= val_transform)
            
            valid_loader = data.DataLoader(valid_set, batch_size=1, shuffle=False)

            return valid_loader

        else:
            test_set = data.Dataset(datalist, transform= test_transform)
            test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False)
            return test_loader
