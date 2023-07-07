import argparse
import config
import torch
import sys
import importlib
import numpy as np
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import lightning_fabric as lf

from engine import Engine
from dataset import JY_Dataset
from models.mobileNet import mobilenet_v3_large, mobilenet_v3_small
from models.efficientNet import efficientNet_b0,efficientNet_b1,efficientNet_b2, efficientNet_b3, efficientNet_v2_s, efficientNet_v2_m
parser = argparse.ArgumentParser() 
parser.add_argument('--mode', type=str, default='train')
args = parser.parse_args()

model_dict = {
        "mobilenet_v3_large" : mobilenet_v3_large(config),
        "mobilenet_v3_small" : mobilenet_v3_small(config),
        "efficientnet_b0" : efficientNet_b0(config),
        "efficientnet_b1" : efficientNet_b1(config),
        "efficientnet_b2" : efficientNet_b2(config),
        "efficientnet_b3" : efficientNet_b3(config),
        "efficientnet_v2_s" : efficientNet_v2_s(config),
        "efficientnet_v2_m" : efficientNet_v2_m(config)
    }

def train() :
    lf.utilities.seed.seed_everything(seed = config.random_seed)
    
    train_dataset = JY_Dataset(dataset = np.load(os.path.join(config.dataset_path, "train.npy"), allow_pickle = True), config = config)
    val_dataset = JY_Dataset(dataset = np.load(os.path.join(config.dataset_path, "val.npy"), allow_pickle = True), config = config)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = model_dict[config.model_name]
    # model = mobilenet_v3_large(config)
    # model = mobilenet_v3_small(config)
    
    engine = Engine(model)
    
    checkpoint_callback = ModelCheckpoint(
    save_top_k=20,
    monitor = "val_loss",
    mode = "min",
    filename = "{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
    )
    
    trainer = pl.Trainer(
        deterministic=False,
        default_root_dir = config.save_path,
        devices = config.num_gpu, 
        val_check_interval = 0.1,
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        callbacks = [checkpoint_callback],
        accelerator = "gpu",
        num_sanity_val_steps = 0,
        resume_from_checkpoint = None, 
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    
    trainer.fit(engine, train_dataloaders = train_dataloader, val_dataloaders =  val_dataloader)
    
    
    
def test() :
    lf.utilities.seed.seed_everything(seed = config.random_seed)
    
    test_dataset = JY_Dataset(dataset = np.load(os.path.join(config.dataset_path, "test.npy"), allow_pickle = True), config = config)
    train_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = model_dict[config.model_name]
    # model = mobilenet_v3_large(config)
    # model = mobilenet_v3_small(config)
    
    # if config.checkpoint_path is not None :
    #     ckpt = torch.load(config.checkpoint_path)
    #     model.load_state_dict(ckpt['state_dict'], strict=False)
    #     print("checkpoint loaded")
    # else :
    #     print("checkpoint not loaded")
    
    engine = Engine(model)
    
    trainer = pl.Trainer(
    deterministic=False,
    devices = config.num_gpu, 
    auto_lr_find = True,    
    sync_batchnorm = True,
    checkpoint_callback = False,
    accelerator = "gpu" ,
    num_sanity_val_steps = 0,
    replace_sampler_ddp = False,
    gradient_clip_val=1.0
    )
    
    if config.checkpoint_path is not None :
        trainer.test(model = engine, test_dataloaders = train_dataloader, ckpt_path = config.checkpoint_path)
        print("checkpoint loaded")
    else :
        print("checkpoint not loaded")
        sys.exit(1)
    
    
if __name__ == "__main__" :
    if args.mode == "train" :
        train()
    elif args.mode == "test" :
        test()
    else :
        print("Wrong mode")
        sys.exit(1)