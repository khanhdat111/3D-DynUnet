from libs.data.hdr2nii import conver_hdr2nii

import json

hdrpath = "/kaggle/input/iseg19/iSeg-2019-Training/iSeg-2019-Training"
niipath = "/kaggle/working/imagesTr"
labelpath = "/kaggle/working/labelsTr"
val_hdrpath = "/kaggle/input/iseg19/iSeg-2019-Validation"
val_niipath = "/kaggle/working/imagesTs"

conver_hdr2nii(hdrpath, niipath, labelpath, val_niipath, val_hdrpath)
with open("/kaggle/working/datalist.json") as f:
    datalist = json.load(f)
    
datalist['training'] = datalist['training']*8
with open('/kaggle/working/datalist.json', "w") as f:
    json.dump(datalist, f)