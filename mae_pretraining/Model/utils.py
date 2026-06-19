import os
import glob
import numpy as np
import math
import torch
from sklearn.metrics import roc_auc_score
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import timm
import time 


# class EarlyStopping:

# def load_pretrained_checkpoint(model, checkpoint_path):
#     files = [f for f in os.listdir(checkpoint_path)]
#     if len(files)>0:
#         files.sort()
#         ckp = files[-1]
#         model.load_state_dict(torch.load(checkpoint_path+ckp)['net'])
#         print(ckp, ' found and loaded.')
    
#     return model

def load_checkpoint(model, filename):
    files = os.listdir(filename)
    if len(files)>0:
        files.sort()
        ckp = files[-1]
        model.load_state_dict(torch.load(filename+ckp)['net'])
        print(ckp, ' found and loaded.')
    
    return model


def load_pretrained_checkpoint(model, pre_trained_model_path, checkpoint_type=None):
    """Loading (transferring) pre-trained MAE model weights

    Parameters
    ----------
    model : torch.nn.Module
        model to finetune
    pre_trained_model_path : str
        path to the pre-trained models checkpoint
    """
    checkpoint_file = glob.glob(pre_trained_model_path + '*')[0]
    
    if pre_trained_model_path == 'imagenet_weights/':
        keys_to_remove = ['head.weight', 'head.bias', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
        # keys_to_remove = ['head.weight', 'head.bias', 'pos_embed']
        # checkpoint_model = timm.create_model('vit_large_patch16_224').state_dict()
        checkpoint_model = timm.create_model('vit_base_patch16_224').state_dict()
        # checkpoint_model = timm.create_model('vit_small_patch16_224').state_dict()
        print('Loaded ImageNet pre-trained checkpoint')
        
    else:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        print("Loaded pre-trained checkpoint from: %s" % checkpoint_file)
        checkpoint_model = checkpoint['net']
        keys_to_remove = ['head.weight', 'head.bias', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']

    state_dict = model.state_dict()
        
    for k in keys_to_remove:
        if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    msg = model.load_state_dict(checkpoint_model, strict=False)
    # print("Missing:", msg.missing_keys)
    # print("Unexpected:", msg.unexpected_keys)
    
    # if checkpoint_type != 'no_pos_embed':
    #     assert set(msg.missing_keys) == set(keys_to_remove), print(msg.missing_keys)

    return model

### The following functions are implemented in the original code but not used anywhere. They are left here for future reference.
def make_scheduler():
    pass

def adjust_alpha():
    pass

def set_requires_grad():
    pass

def loop_iterable(iterable):
    pass

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        val_loss = val_loss / 100
        if self.best_loss == None:
            self.best_loss = val_loss
        elif val_loss - self.best_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif val_loss - self.best_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def save_model(args, cfg, model, filename, epoch, steps, finetune=False):
    flist = glob.glob(filename+ '*')
    for f in flist:
        os.remove(f)
    
    plane = args.plane
    today = time.time()
    if finetune:
        filename = f"{filename}_epoch{epoch+1}_steps{steps}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth.tar"
    else:
        mask_ratio = args.mask_ratio
        filename = f"{filename}_epoch{epoch+1}_steps{steps}_plane{plane}_mask{mask_ratio:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth.tar"
    if len([x for x in args.devices.split(",")]) > 1:
        state = {"net": model.module.state_dict()}
    else:
        state = {"net": model.state_dict()}
    torch.save(state, filename)

def adjust_learning_rate_halfcosine(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg['SOLVER']['warmup_epochs']:
        lr = cfg['SOLVER']['lr'] * epoch / cfg['SOLVER']['warmup_epochs'] 
    else:
        lr = cfg['SOLVER']['min_lr'] + (cfg['SOLVER']['lr'] - cfg['SOLVER']['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - cfg['SOLVER']['warmup_epochs']) / (cfg['TRAINING']['EPOCHS'] - cfg['SOLVER']['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, dist_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if dist_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), np.zeros([1, embed_dim]), pos_embed], axis=0)
    elif cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode each axis
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb