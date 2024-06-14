import os
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import monai
import json

monai.utils.set_determinism(seed=123)

def process_brats_files(training_datapath, validation_datapath, niipath, labelpath, val_niipath):

    os.makedirs(niipath, exist_ok=True)
    os.makedirs(labelpath, exist_ok=True)
    os.makedirs(val_niipath, exist_ok=True)
    
    train, val = [], []

    num_train_subjects = 369  
    for i in range(1, num_train_subjects + 1):
        subject_id = f'BraTS20_Training_{i:03d}'
        subject_path = os.path.join(training_datapath, subject_id)
        flair_path = os.path.join(subject_path, f'{subject_id}_flair.nii')
        t1_path = os.path.join(subject_path, f'{subject_id}_t1.nii')
        t1ce_path = os.path.join(subject_path, f'{subject_id}_t1ce.nii')
        t2_path = os.path.join(subject_path, f'{subject_id}_t2.nii')
        if i==355:
            seg_path = os.path.join(subject_path, 'W39_1998.09.19_Segm.nii')
        else:
            seg_path = os.path.join(subject_path, f'{subject_id}_seg.nii')
        train.append({'label': [seg_path], 'image': [flair_path, t1_path, t1ce_path, t2_path]})

    num_val_subjects = 125  
    for i in range(1, num_val_subjects + 1):
        subject_id = f'BraTS20_Training_{i:03d}'
        subject_path = os.path.join(validation_datapath, subject_id)
        flair_path = os.path.join(subject_path, f'{subject_id}_flair.nii')
        t1_path = os.path.join(subject_path, f'{subject_id}_t1.nii')
        t1ce_path = os.path.join(subject_path, f'{subject_id}_t1ce.nii')
        t2_path = os.path.join(subject_path, f'{subject_id}_t2.nii')

        val.append({'image': [flair_path, t1_path, t1ce_path, t2_path]})

    train_list, val_list = train_test_split(train, train_size=0.8)
    datalist = {"training": train, "validation": val_list, "testing": train}

    with open('/kaggle/working/datalist.json', "w") as f:
        json.dump(datalist, f)
