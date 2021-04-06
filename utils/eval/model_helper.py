import torch
import os
import numpy as np
from argparse import Namespace

from utils.common.setup_helper import load_weights
from utils.datasets.preprocess import load_im_flexible, load_im_tensor
from networks.patch2pix import Patch2Pix

def init_patch2pix_matcher(args):  
    net = load_model(args.ckpt, method='patch2pix')
    matcher = lambda imq, imr: estimate_matches(net, imq, imr,
                                                ksize=args.ksize, 
                                                io_thres=args.io_thres,
                                                eval_type='fine', 
                                                imsize=args.imsize)
    return matcher

def init_ncn_matcher(args):  
    net = load_model(args.ckpt, method='nc')
    matcher = lambda imq, imr: estimate_matches(net, imq, imr, 
                                                ksize=args.ksize, 
                                                ncn_thres=args.ncn_thres,
                                                eval_type='coarse', 
                                                imsize=args.imsize)
    return matcher
    
def load_model(ckpt_path, method='patch2pix', lprint=print):    
    # Initialize network
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu') 
    ckpt = load_weights(ckpt_path, device)    
    config = Namespace(training=False,
                       device=device,
                       regr_batch=1200,
                       backbone='ResNet34',
                       feat_idx=None,
                       weights_dict=None,
                       regressor_config=None,
                       change_stride=True)
    lprint('\nLoad model method:{} '.format(method))
    if 'patch2pix' in method:
        config.backbone = ckpt['backbone']
        config.feat_idx = ckpt['feat_idx']
        config.weights_dict = ckpt['state_dict']
        config.regressor_config = ckpt['regressor_config']
        config.regressor_config.panc = 1 # Only use panc 1 during evaluation        
        if 'last_epoch' in ckpt:
            epoch = ckpt['last_epoch']+1
            lprint(f'Ckpt:{ckpt_path} epochs:{epoch}')
        else:
            lprint(f'Ckpt:{ckpt_path}')
            
    elif 'nc' in method:
        if type(ckpt) is dict:
            ckpt = ckpt['state_dict']
        config.weights_dict = ckpt
        lprint('Load pretrained weights: {}'.format(ckpt_path))
    else:
        lprint('Wrong method name.')
    net = Patch2Pix(config)
    net.eval()
    return net

def estimate_matches(net, im1, im2, ksize=2, ncn_thres=0.0, mutual=True, 
                     io_thres=0.25, eval_type='fine', imsize=None):
    # Assume batch size is 1
    # Load images
    im1, sc1 = load_im_flexible(im1, ksize, net.upsample, imsize=imsize)
    im2, sc2 = load_im_flexible(im2, ksize, net.upsample, imsize=imsize)
    upscale = np.array([sc1 + sc2])
    im1 = im1.unsqueeze(0).to(net.device)
    im2 = im2.unsqueeze(0).to(net.device)
    
    # Coarse matching
    if eval_type == 'coarse':    
        with torch.no_grad():
            coarse_matches, scores = net.predict_coarse(im1, im2, ksize=ksize,
                                                        ncn_thres=ncn_thres, 
                                                        mutual=mutual)
        matches = coarse_matches[0].cpu().data.numpy()
        scores = scores[0].cpu().data.numpy()        
        matches = upscale * matches        
        return matches, scores, matches
   
    # Fine matching
    if eval_type == 'fine':
        # Fine matches
        with torch.no_grad():
            fine_matches, fine_scores, coarse_matches = net.predict_fine(im1, im2, ksize=ksize, 
                                                                         ncn_thres=ncn_thres, 
                                                                         mutual=mutual)
            coarse_matches = coarse_matches[0].cpu().data.numpy()
            fine_matches = fine_matches[0].cpu().data.numpy()
            fine_scores = fine_scores[0].cpu().data.numpy()
             
    # Inlier filtering
    pos_ids = np.where(fine_scores > io_thres)[0]
    if len(pos_ids) > 0:
        coarse_matches = coarse_matches[pos_ids]
        matches = fine_matches[pos_ids]
        scores = fine_scores[pos_ids]
    else:
        # Simply take all matches for this case
        matches = fine_matches
        scores = fine_scores         

    matches = upscale * matches
    coarse_matches = upscale * coarse_matches    
    return matches, scores, coarse_matches

def refine_matches(im1_path, im2_path, net, coarse_matcher, 
                   io_thres=0.0, imsize=None, coarse_only=False):
    # Load images
    im1, grey1, sc1 = load_im_tensor(im1_path, net.device, imsize, with_gray=True)
    im2, grey2, sc2 = load_im_tensor(im2_path, net.device, imsize, with_gray=True)
    upscale = np.array([sc1 + sc2])

    # Predict coarse matches 
    coarse_matches = coarse_matcher(grey1, grey2)    
    if coarse_only:
        coarse_all = upscale * coarse_matches.cpu().data.numpy()    
        return coarse_all, None, None
    
    refined_matches, scores, coarse_matches = net.refine_matches(im1, im2, coarse_matches, io_thres)
    refined_matches = upscale * refined_matches
    coarse_matches = upscale * coarse_matches
    return refined_matches, scores, coarse_matches                   

                   
