import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

L2Normalize = lambda feat, dim: feat / torch.pow(torch.sum(torch.pow(feat, 2), dim=dim) + 1e-6, 0.5).unsqueeze(dim)

def cal_conv_out_size(w, kernel_size, stride, padding):
    return (w - kernel_size + 2 * padding) // stride + 1

def maxpool4d(corr4d_hres, k_size=4):
    slices=[]
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                for l in range(k_size):
                    sl = corr4d_hres[:,:,i::k_size,j::k_size,k::k_size,l::k_size] # Support batches
                    slices.append(sl)
                    
    slices = torch.cat(tuple(slices),dim=1)  # B, ksize*4, h1, w1, h2, w2
    corr4d, max_idx = torch.max(slices,dim=1,keepdim=True)
    
    # i,j,k,l represent the *relative* coords of the max point in the box of size k_size*k_size*k_size*k_size    
    if torch.__version__ >= '1.6.0':
        max_l=torch.fmod(max_idx,k_size)
        max_k=torch.fmod(max_idx.sub(max_l).floor_divide(k_size),k_size)
        max_j=torch.fmod(max_idx.sub(max_l).floor_divide(k_size).sub(max_k).floor_divide(k_size),k_size)
        max_i=max_idx.sub(max_l).floor_divide(k_size).sub(max_k).floor_divide(k_size).sub(max_j).floor_divide(k_size)
    else:
        max_l=torch.fmod(max_idx,k_size)
        max_k=torch.fmod(max_idx.sub(max_l).div(k_size),k_size)
        max_j=torch.fmod(max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size),k_size)
        max_i=max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size).sub(max_j).div(k_size)
    return (corr4d,max_i,max_j,max_k,max_l)

class FeatCorrelation(torch.nn.Module):
    def __init__(self, shape='4D'):
        super().__init__()
        self.shape = shape
    
    def forward(self, feat1, feat2):        
        b, c, h1, w1 = feat1.size()
        b, c, h2, w2 = feat2.size()
        feat1 = feat1.view(b, c, h1*w1).transpose(1, 2) # size [b, h1*w1, c]
        feat2 = feat2.view(b, c, h2*w2)  # size [b, c, h2*w2]
        
        # Matrix multiplication
        correlation = torch.bmm(feat1, feat2)  # [b, h1*w1, h2*w2]
        if self.shape == '3D':
            correlation = correlation.view(b, h1, w1, h2*w2).permute(0, 3, 1, 2)  # [b, h2*w2, h1, w1]            
        elif self.shape == '4D':
            correlation = correlation.view(b, h1, w1, h2, w2).unsqueeze(1) # [b, 1, h1, w1, h2, w2]
        return correlation

    
class FeatRegressNet(nn.Module):
    def __init__(self, config, psize=16, out_dim=5):
        super().__init__()
        self.psize = psize
        self.conv_strs = config.conv_strs if 'conv_strs' in config else [2] * len(config.conv_kers)
        self.conv_dims = config.conv_dims
        self.conv_kers = config.conv_kers
        self.feat_comb = config.feat_comb  # Combine 2 feature maps before the conv or after the conv
        self.feat_dim = config.feat_dim if self.feat_comb == 'post' else 2 * config.feat_dim
        self.fc_in_dim = config.conv_dims[-1] * 2 if self.feat_comb == 'post' else config.conv_dims[-1]
        
        # Build layers    
        self.conv = self.make_conv_layers(self.feat_dim, self.conv_dims, self.conv_kers)
        self.fc = self.make_fc_layers(self.fc_in_dim, config.fc_dims, out_dim)        
        print(f'FeatRegressNet:  feat_comb:{self.feat_comb} ' \
              f'psize:{self.psize} out:{out_dim} ' \
              f'feat_dim:{self.feat_dim} conv_kers:{self.conv_kers} ' \
              f'conv_dims:{self.conv_dims} conv_str:{self.conv_strs} ' 
              )        
        
    def make_conv_layers(self, in_dim, conv_dims, conv_kers, bias=False):
        layers = []
        w = self.psize  # Initial spatial size        
        for out_dim, kernel_size, stride in zip(conv_dims, conv_kers, self.conv_strs):
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=1, bias=bias))
            layers.append(nn.BatchNorm2d(out_dim))
            w = cal_conv_out_size(w, kernel_size, stride, 1)
            in_dim = out_dim
        layers.append(nn.ReLU())        
        # To make sure spatial dim goes to 1, one can also use AdaptiveMaxPool
        layers.append(nn.MaxPool2d(kernel_size=w))
        return nn.Sequential(*layers)
        
    def make_fc_layers(self, in_dim, fc_dims, fc_out_dim):
        layers = []
        for out_dim in fc_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim)),
            layers.append(nn.ReLU())            
            in_dim = out_dim
            
        # Final layer
        layers.append(nn.Linear(in_dim, fc_out_dim))
        return nn.Sequential(*layers)
    
    def forward(self, feat1, feat2):
        # feat1, feat2: shape (N, D, 16, 16)       
        if self.feat_comb == 'pre':
            feat = torch.cat([feat1, feat2], dim=1)            
            feat = self.conv(feat)  # N, D, 1, 1
        else:
            feat1 = self.conv(feat1)
            feat2 = self.conv(feat2)            
            feat = torch.cat([feat1, feat2], dim=1)  # N, D, 1, 1
        feat = feat.view(-1, feat.shape[1])
        out = self.fc(feat)  # N, 5
        return out 
    
def init_optimizer(params, config):
    if config.opt == 'adam':
        optimizer = torch.optim.Adam(params, lr=config.lr_init, weight_decay=config.weight_decay)
        print('Setup  Adam optimizer(lr={},wd={})'.format(config.lr_init, config.weight_decay))

    elif config.opt == 'sgd':
        optimizer = torch.optim.SGD(params, momentum=0.9, lr=config.lr_init, weight_decay=config.weight_decay)
        print('Setup  SGD optimizer(lr={},wd={},mom=0.9)'.format(config.lr_init, config.weight_decay))

    if config.optimizer_dict:
        optimizer.load_state_dict(config.optimizer_dict)

    # Schedule learning rate decay  lr_decay = ['name', params] or None
    lr_scheduler = None
    if 'lr_decay' in config and config.lr_decay:
        if config.lr_decay[0] == 'step':
            decay_factor, decay_step = float(config.lr_decay[1]), int(config.lr_decay[2])
            last_epoch = config.start_epoch - 1
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                           step_size=decay_step, 
                                                           gamma=decay_factor, 
                                                           last_epoch=last_epoch)
            print(f'Setup StepLR Decay: decay_factor={decay_factor} '
                  f'step={decay_step} last_epoch={last_epoch}')

        elif config.lr_decay[0] == 'multistep':
            decay_factor = float(config.lr_decay[1])
            decay_steps = [int(v) for v in config.lr_decay[2::]]
            last_epoch = config.start_epoch - 1
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=decay_steps,
                                                                gamma=decay_factor, 
                                                                last_epoch=last_epoch)
            print(f'Setup MultiStepLR Decay: decay_factor={decay_factor} '
                  f'steps={decay_steps} last_epoch={last_epoch}')

        if config.lr_scheduler_dict and lr_scheduler:
            lr_scheduler.load_state_dict(config.lr_scheduler_dict)
    return optimizer, lr_scheduler     
    
def xavier_init_func_(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:  # Incase bias is turned off            
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:  # Incase bias is turned off            
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    