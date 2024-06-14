# BrainTumour_Seg

## How to install:
1. Open Kaggle and add this dataset:
```
https://www.kaggle.com/datasets/rcshoo/iseg19
```
2. Clone the repo and install the dependent libraries
```python
get_ipython().system(f"cd /kaggle/working/")
GITHUB_TOKEN = "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"
USER = "RC-Sho0"
CLONE_URL = f"https://{USER}:{GITHUB_TOKEN}@github.com/{USER}/BrainTumour_Seg.git"
get_ipython().system(f"git clone -b icta {CLONE_URL}")
%cd /kaggle/working/BrainTumour_Seg
get_ipython().system(f"python utils/setup.py <wandb key>")
get_ipython().system(f"python preprocess_data.py") ##For iseg dataset
```

3. Adjust parameters in file config.json
```json
{
    "model_name": "", 
    "att": null,
    "in_channel": 2,
    "out_channel": 4,
    "project": "iseg",
    "model_trained": null,
    "datalist": "/kaggle/working/datalist.json",
    "config":{
        "step_val": 5,
        "loss": "dice",
        "max_epochs": 100,
        "name":"dynunet_dda_s1",
        "lr":3e-4,
        "tmax": 100,
        "results_dir":"/kaggle/working/results",
        "log": true
    }
} 
```
    - Model name: ['dynunet','dynunet_dda','segresnet','swinunetr','vnet','dynunet_cbam','dsdynunet','dsdynunet_cbam','dsdynunet_dda']
    - Att: number of filter add Attention (exp: [32,64] for DDA in stage 1,2 in dynunet_dda model)
    - in_channel: number channel input
    - out_channel: number channel output
    - project: wandb project name
    - model_trained: weighted of model (optinal for pretrained)
    - datalist: path of datalist
    - config: 
        + step_val: number of epochs model skip run valid step
        + max_epochs: max epochs
        + name: name of experiment on wanbd
        + lr: init learning rate 
        + log: let it true if you want to track on wandb


4. Runnn:
```
!python seg_train.py --input /kaggle/working/exp.json
```
