import torch
import os
import numpy as np
from argparse import Namespace
import visdom
from utils.common.visdom_helper import VisPlots
from utils.common.setup_helper import *

def save_ckpt(net, epoch, sav_dir, best_vals=None, last_ckpt=False, is_best=False, name=None):
    ckpt = {'last_epoch': epoch,
            'best_vals' : best_vals,
            'backbone' : net.backbone,     
            'feat_idx' : net.feat_idx,
            'change_stride' : net.change_stride,
            'regressor_config' : net.regressor_config,
            'state_dict': net.state_dict(),
            'optim': net.optimizer.state_dict()}
    
    if net.lr_scheduler:
        ckpt['lr_scheduler'] = net.lr_scheduler.state_dict()
    
    if last_ckpt:
        ckpt_name = 'last_ckpt.pth' 
    elif is_best:
        ckpt_name = 'best_ckpt.pth' 
    else:    
        ckpt_name = 'ckpt_ep{}.pth'.format(epoch+1)
    
    # Overwrite name
    if name:
        ckpt_name = '{}.pth'.format(name)
    ckpt_path = os.path.join(sav_dir, ckpt_name)
    torch.save(ckpt, ckpt_path)

def load_ckpt(ckpt_path, config, resume=False):    
    # Fine matching dist and io , qt_err, pass_rate 
    best_vals = [np.inf, 0.0, np.inf, 0.0]
    ckpt = load_weights(ckpt_path, config.device)
    
    if 'backbone' in ckpt:
        config.feat_idx = ckpt['feat_idx']
        config.weights_dict = ckpt['state_dict']
        config.backbone = ckpt['backbone']
        config.regressor_config = ckpt['regressor_config']
        if 'change_stride' in ckpt:
            config.change_stride = ckpt['change_stride']        
                
        if resume:
            config.start_epoch = ckpt['last_epoch'] + 1            
            config.optim_config.optimizer_dict = ckpt['optim']
            if 'lr_scheduler' in ckpt:
                config.optim_config.lr_scheduler_dict = ckpt['lr_scheduler']  

            if 'best_vals' in ckpt:
                if len(ckpt['best_vals']) == len(best_vals):
                    best_vals = ckpt['best_vals']
    
    else:
        # Only the pretrained weights
        config.weights_dict = ckpt        
    return best_vals


def init_model_config(args, lprint_):
    """This is a quick wrapper for model initialization
    Currently support method = patch2pix / ncnet.
    """
    
     # Initialize model
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')    
    regressor_config = Namespace(conv_dims=args.conv_dims,
                                 conv_kers=args.conv_kers,
                                 conv_strs=args.conv_strs,
                                 fc_dims=args.fc_dims,
                                 feat_comb=args.feat_comb,
                                 psize=args.psize,
                                 pshift = args.pshift,
                                 panc = args.panc,
                                 shared = args.shared)
    
    optim_config = Namespace(opt='adam',
                             lr_init=args.lr_init,
                             weight_decay=args.weight_decay,
                             lr_decay=args.lr_decay,
                             optimizer_dict=None,
                             lr_scheduler_dict=None)
    
    config = Namespace(training=True,
                       start_epoch=0,
                       device=device,
                       regr_batch=args.regr_batch,
                       backbone=args.backbone,
                       freeze_feat=args.freeze_feat,
                       change_stride=args.change_stride,
                       feat_idx=args.feat_idx,
                       regressor_config=regressor_config, 
                       weights_dict=None,
                       optim_config=optim_config) 
        
    
    # Fine matching dist and io , qt_err, pass_rate
    best_vals = [np.inf, 0.0, np.inf, 0.0]    
    if args.resume:
        # Continue training
        ckpt = os.path.join(args.out_dir, 'last_ckpt.pth')
        if os.path.exists(ckpt):
            args.ckpt = ckpt
    if args.pretrain:
        # Initialize with pretrained nc
        best_vals = load_ckpt(args.pretrain, config)
        lprint_('Load pretrained: {}  vals: {}'.format(args.pretrain, best_vals))        
    if args.ckpt:
        # Load a specific model
        best_vals = load_ckpt(args.ckpt, config, resume=args.resume)
        lprint_('Load model: {}  vals: {}'.format(args.ckpt, best_vals))
        
    return config, best_vals

def get_visdom_plots(prefix='train', env='main', server='localhost', port=9333):
    """Initialize Visdom plots following the pre-defined schema.
    Adapt train_plots Namespace if one needs to add/remove legends or plots.
    Args:
        - prefix: the name prefix will be add to the original name of each plot.
        - env: the name of visdom envirionment where plots will appear there.
        - server: the name of the host where visdom server is running. 
                Make sure visdom service is running correctly on specified port and host.
                No visdom connection will be initialized if server is None. 
                And the program gives dummy plots.
    """
    if server is None:
        vis = None
    else:
        
        vis = visdom.Visdom(server='http://{}'.format(server), port=port)
    
    """Initialize visdom plots: 
    plots format:  Namespace([plot_name]=Namespace([plot_legends...]))    
    """
    plots = Namespace(pair_scores=Namespace(pos=None, neg=None), 
                      cls_ratios=Namespace(mpos_gt=None, mpos_pred=None, 
                                           fpos_gt=None, fpos_pred=None),
                      loss=Namespace(pair=None, nc=None, 
                                     cls_mid=None, cls_fine=None, 
                                     epi_mid=None, epi_fine=None),
                      cls_mid=Namespace(rec=None, prec=None, spec=None, acc=None, f1=None),
                      cls_fine=Namespace(rec=None, prec=None, spec=None, acc=None, f1=None),
                      match_dist=Namespace(cmid_gt=None, mmid_gt=None, 
                                           mfid_gt=None, ffid_gt=None,
                                           cmid_pred=None, mmid_pred=None,
                                           mfid_pred=None, ffid_pred=None),
                      mem=Namespace(rss=None, vms=None, 
                                    gpu_maloc=None, 
                                    gpu_mres=None))
    vis_plots = VisPlots(plots, vis, env=env, prefix=prefix)
    return vis_plots

def plot_cls_metric(mpred, mgt, thres=0.5, plot=None):
    """
    Args:
        - mpred: predicted probability(mask), torch tensor with shape (N,)
        - mgt: ground truth probability(mask), same shape as pred. 
    """
    if not plot:
        return 
    
    try:
        Pgt = mgt
        Ngt = (mgt == 0).float()
        Pgt_num = Pgt.sum()
        Ngt_num = Ngt.sum() 

        Ppred = (mpred > thres).float()
        Npred = (mpred <= thres).float()                        
        TP = (Ppred * Pgt).sum()
        TN = (Npred * Ngt).sum()

        recall = (TP / Pgt_num).item() if Pgt_num > 0 else (1.0 if Ppred.sum() == 0.0 else 0) # Correct pred pos among all gt pos
        specifity = (TN / Ngt_num).item() if Ngt_num > 0 else (1.0 if Npred.sum() == 0.0 else 0) # Correct pred neg among all gt neg           
        precision = (TP / Ppred.sum()).item() if Ppred.sum() > 0 else 0 # Correct pos among predicted pos
        accuracy = (Pgt == Ppred).float().mean().item() # Correct preds among all preds
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Update the plot
        plot.rec.append(recall)
        plot.spec.append(specifity)        
        plot.prec.append(precision)
        plot.acc.append(accuracy)
        plot.f1.append(f1)        
    except:
        print('Error happened during plot cls')
        print(f'mpred={mpred.shape} mgt={mgt.shape}')
        print(f'Pgt_num={Pgt_num} Ngt_num={Ngt_num}')
        return 
        
    return Namespace(recall=recall, specifity=specifity, precision=precision, accuracy=accuracy, f1=f1)

