# -*- coding: utf-8 -*-
"""
"""


from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
# from skimage import io, color, segmentation

import os
import glob
from datetime import datetime
import argparse
import random
import yaml
import logging
import wandb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda import amp
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

from dataset_finetune import ADNIDataset,ADNIDataModule, get_train_transform, get_val_transform
from train_inference import do_train, do_inference
from Model.utils import EarlyStopping, load_pretrained_checkpoint
from Model.build_model import build_ViT, make_optimizer

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")
    
    
def wandb_setup(cfg, args, SAVE_DIR):
    # start a new wandb run to track this script
    os.makedirs(SAVE_DIR+'/'+'./wandb/', exist_ok=True)
    wandb.init(
        # set the wandb project where this run will be logged
        project="MAE Pre-training",
        name=SAVE_DIR,
        
        # track hyperparameters and run metadata
        config={
        "lr": cfg['SOLVER']['lr'],
        "dir": SAVE_DIR+'/'+'./wandb/',
        #"mask ratio": args.mask_ratio,
        "seed": args.seed
        }
    )


def setup_logger(SAVE_DIR, timestamp_current):
    # Set up the logger
    os.makedirs(SAVE_DIR +'/logs/', exist_ok=True)
    if len(glob.glob( SAVE_DIR +'/logs/'+ '*')) > 0:
        print('Logger found...')
        print('----------------')
        logging.basicConfig(filename=glob.glob(SAVE_DIR + '/logs/' + '*')[-1],
                            format='%(asctime)s %(message)s',
                            filemode='a',
                            level=logging.DEBUG, 
                            force=True)
    else:
        logging.basicConfig(filename=SAVE_DIR + '/logs/' + timestamp_current + '.log', 
                        format='%(asctime)s %(message)s',
                        filemode='w',
                        level=logging.DEBUG, 
                        force=True)
    logger = logging.getLogger()
    return logger



if __name__ == '__main__':

    # Parse some variable configs
    parser = argparse.ArgumentParser(description='Fine-tune model for MRI imaging for classification of AD')
    parser.add_argument('--plane', required=True, type=str, choices=['sagittal', 'coronal', 'axial', 'all'], help='Input plane for the model')
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4],
        help="Fold index (0-based) for 5-fold cross-validation.",
    )
    parser.add_argument('--seed',default=42, type=int, help='Experiment seed (for reproducible results)')
    parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count of training')
    parser.add_argument('--checkpoint', default='Results/Pretraining/__seed_42/Checkpoints/', type=str, help='Checkpoint model path')
    parser.add_argument('--save_dir', default='Results/Finetuning/', type=str, help='Directory to save trained model')
    parser.add_argument('--data_dir', default='Data/', type=str, help='Directory containing the fMRI data')
    
    args = parser.parse_args()

    base_directory = "/home/hpc/iwi5/iwi5360h/ADMRI/fmriAD/mae_pretraining/"
    config_file = open(base_directory + "config.yml", 'rb')
    cfg = yaml.load(config_file, Loader=yaml.FullLoader)
    
    
    SAVE_DIR = base_directory + args.save_dir + '_' +  '_seed_' + str(args.seed)
    DATA_DIR = base_directory + args.data_dir
    checkpoint_pretrained = base_directory + args.checkpoint
    checkpoint_finetune = SAVE_DIR + '/' + 'Checkpoints/'
    os.makedirs(checkpoint_finetune, exist_ok=True)
    
    
    
    
    
    # Set seed
    set_seed(args.seed)

    timestamp_current = datetime.now()
    timestamp_current = timestamp_current.strftime("%Y%m%d_%H%M")

    # Set up logger file
    logger = setup_logger(SAVE_DIR, timestamp_current)
    logger.setLevel(logging.DEBUG)
    logger.info('Process number: %d'%(os.getpid()))
    logger.info("Started training. Savename : " + args.save_dir)
    logger.info("Seed : " + str(args.seed))
    
    model = build_ViT(cfg, args)
    model = load_pretrained_checkpoint(model, checkpoint_pretrained)
    model.cuda()

    optimizer = make_optimizer(cfg, args, model)
    scaler = amp.GradScaler()
    
    
    
    # Early Stopper
    early_stopper = None
    if cfg['TRAINING']['EARLY_STOPPING']:
        early_stopper = EarlyStopping(patience=cfg['TRAINING']['EARLY_STOPPING_PATIENCE'], 
                                      min_delta=cfg['TRAINING']['EARLY_STOPPING_DELTA'])        
    
    # Save all configs and args just in case
    logger.info(cfg)
    logger.info(args)

    # Init wandb
    wandb_setup(cfg, args, SAVE_DIR)
    
    # Build Datasets and Dataloaders
    dm = ADNIDataModule(
        plane=args.plane,
        fold=args.fold,
        batch_size=cfg["DATALOADER"]["BATCH_SIZE"],
        num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
    )
    dm.setup()
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()
    train_dataset = train_dataloader.dataset
    val_dataset = val_dataloader.dataset
    test_dataset = test_dataloader.dataset
    
    
    # Initialize loss function and optimizer
    weight_balance = train_dataset.get_class_weights().cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_balance)
    

    trained_model = do_train(
        cfg, args, checkpoint_finetune, model, criterion, optimizer, scaler, train_dataloader,
        train_dataset, logger, early_stopper, True,
        val_dataloader
    )
    

    test_acc, bal_acc, corrects, n_datapoints = do_inference(
            cfg, args, trained_model, test_dataloader, logger
        )




    





