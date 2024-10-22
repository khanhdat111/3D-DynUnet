import os
import tempfile
import monai
from monai import transforms
from monai import data
import numpy as np
import torch
from monai.transforms import Transform, MapTransform


class Iseg(transforms.MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 1.0)
            result.append(d[key] == 2.0)
            result.append(d[key] == 3.0)
            d[key] = torch.stack(result, axis=0).float()
        return d


def iseg_dataloader(datalist, batch_size, stage, shuffle):
    
        train_transform = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image", "label"]),
                    transforms.EnsureChannelFirstd(keys="image"),
                    transforms.EnsureTyped(keys=["image", "label"]),
                    Iseg(keys='label'),
                    transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                    transforms.Spacingd(
                        keys=["image", "label"],
                        pixdim=(1.0, 1.0, 1.0),
                        mode=("bilinear", "nearest"),
                    ),
                    transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),
                    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                    transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                ]
            )

        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.EnsureTyped(keys=["image", "label"]),
                Iseg(keys='label'),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )        
    

        if stage == 'train':
            train_set = data.Dataset(datalist['training'], transform= train_transform)
            valid_set = data.Dataset(datalist['validation'], transform= val_transform)

            train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
            valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
            
            return train_loader, valid_loader

        else:
            test_set = data.Dataset(datalist, transform= val_transform)
            test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
            return test_loader
