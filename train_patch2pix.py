import argparse
from argparse import Namespace
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from utils.datasets import ImMatchDatasetMega
from utils.train.helper import *
from utils.train.eval_epoch_immatch import eval_immatch_val_sets
from utils.common.setup_helper import *
from networks.utils import sampson_dist, filter_coarse
from networks.patch2pix import Patch2Pix

def parse_agrs():
    parser = argparse.ArgumentParser(description='Train Patch2Pix Matching Network')
    parser.add_argument('--gpu', '-gpu', type=int, default=0) 
    parser.add_argument('--seed',  type=int, default=1)     
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--save_step', type=int, default=1) 
    parser.add_argument('--plot_counts', type=int, default=5)    
    parser.add_argument('--batch', type=int, default=8) 
    parser.add_argument('--regr_batch', type=int, default=1200)
    parser.add_argument('--visdom_host', '-vh',  type=str, default=None)  
    parser.add_argument('--visdom_port', '-vp',  type=str, default=None)      
    parser.add_argument('--prefix', type=str, default='')    
    parser.add_argument('--out_dir', '-o', type=str, default='output/patch2pix')  
    
    # Data loading config
    parser.add_argument('--dataset', type=str, default='MegaDepth')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--pair_root', type=str, default='data_pairs')
    parser.add_argument(
        '--match_npy', type=str,                     
        default='megadepth_pairs.ov0.35_imrat1.5.pair500.excl_test.npy'
    )
    
    # Model architecture
    parser.add_argument('--backbone', type=str, default='ResNet34')
    parser.add_argument('--change_stride', action='store_true')
    parser.add_argument('--ksize', type=int, default=2)
    parser.add_argument('--freeze_feat', type=int, default=87)    
    parser.add_argument('--feat_idx', type=int, nargs='*', default=[0, 1, 2, 3])    
    parser.add_argument('--feat_comb', type=str, default='pre')
    parser.add_argument('--conv_kers', type=int, nargs='*', default=[3, 3])
    parser.add_argument('--conv_dims', type=int, nargs='*', default=[512, 512])
    parser.add_argument('--conv_strs', type=int, nargs='*', default=[2, 1])
    parser.add_argument('--fc_dims', type=int, nargs='*', default=[512, 256])    
    parser.add_argument('--psize', type=int, nargs=2, default=[16, 16])
    parser.add_argument('--pshift', type=int, default=8)
    parser.add_argument('--panc', type=int, choices=[8, 1], default=8)    
    parser.add_argument('--ptmax', type=int, default=400)    
    parser.add_argument('--shared', action='store_true')    
    
    # Matching thresholds
    parser.add_argument('--cthres', type=float, default=0.5)
    parser.add_argument('--cls_dthres', type=int, nargs=2, default=[50, 5])
    parser.add_argument('--epi_dthres', type=int, nargs=2, default=[50, 5])

    # Model intialize
    parser.add_argument('--pretrain', type=str, default=None)   
    parser.add_argument('--ckpt', type=str, default=None)                            
    parser.add_argument('--resume', action='store_true')  # Auto load last cpkt
        
    # Optimization
    parser.add_argument('--lr_init', '-lr', metavar='%f', type=float, default=5e-4) 
    parser.add_argument('--lr_decay', '-lrd', metavar='%s[type] %f[*factor] %d[*step]', nargs='*', default=None) # Opt: 'step' 'multistep'
    parser.add_argument('--weight_decay', '-wd', metavar='%f', type=float, default=0) 
    parser.add_argument('--weight_cls', '-wcls', metavar='%f', type=float, default=10.0)
    parser.add_argument('--weight_epi', '-wepi', metavar='%f[fine] %f[mid]', type=float, nargs='*', default=[1, 1])

    args = parser.parse_args()
    return args
    
def train_epoch(epoch, net, train_loader, train_vis, args, lprint_):    
    net.train()
    train_vis.clear()  # Clearn visdom plot data per epoch    
    plot_step = len(train_loader) // args.plot_counts
    
    # Setup threshold params
    net.panc = args.panc
    ksize = args.ksize
    cthres, cls_dthres, epi_dthres = args.cthres, args.cls_dthres, args.epi_dthres
    cls_loss_weight = args.weight_cls
    efine_weight, emid_weight = args.weight_epi

    # Start training
    skipped = 0
    lprint_(f'ksize={ksize} cthres={cthres} cls_dthres={cls_dthres} '
            f'epi_dthre={epi_dthres} ptmax={args.ptmax} panc={net.panc}')
    for i, batch in enumerate(train_loader):
        im_src, im_pos, Fs = net.load_batch_(batch, dtype='pair')
        
        # Estimate patch-level matches 
        corr4d, delta4d, feats1, feats2 = net.forward(im_src, im_pos, ksize=ksize, return_feats=True)
        coarse_matches, match_scores = net.cal_coarse_matches(corr4d, delta4d, ksize=ksize, upsample=net.upsample, center=True)
        
        if net.panc > 1 and args.ptmax > 0:
            coarse_matches, match_scores = filter_coarse(coarse_matches, match_scores, 0.0, True, ptmax=args.ptmax)        

        # Coarse matches to locate anchors    
        coarse_matches = net.shift_to_anchors(coarse_matches)

        # Mid level matching on positive pairs
        mid_matches, mid_probs = net.forward_fine_match(feats1, feats2, 
                                                        coarse_matches, 
                                                        psize=net.psize[0],
                                                        ptype=net.ptype[0],
                                                        regressor=net.regress_mid)
        
        # Fine level matching based on mid matches
        fine_matches, fine_probs = net.forward_fine_match(feats1, feats2, 
                                                          mid_matches,               
                                                          psize=net.psize[1],  
                                                          ptype=net.ptype[1],
                                                          regressor=net.regress_fine)
        # Calculate per pair losses
        cls_batch_lss = []
        epi_batch_lss = []
        for F, cmat, mmat, fmat, mcls_pred, fcls_pred in zip(Fs, coarse_matches, 
                                                             mid_matches, fine_matches, 
                                                             mid_probs, fine_probs):
            N = len(cmat)
            
            # Classification gt based on coarse matches             
            cdist = net.geo_dist_fn(cmat, F)
            mdist = net.geo_dist_fn(mmat, F)
            fdist = net.geo_dist_fn(fmat, F)
            ones = torch.ones_like(cdist)
            zeros = torch.zeros_like(cdist)

            # Classification loss
            mcls_pos = torch.where(cdist < cls_dthres[0], ones, zeros)
            fcls_pos = torch.where(mdist < cls_dthres[1], ones, zeros)
            mcls_neg = 1 - mcls_pos
            fcls_neg = 1 - fcls_pos
            
            if mcls_pos.sum() == 0 or fcls_pos.sum() == 0:
                skipped += 1
                continue
                        
            mcls_weights = mcls_neg.sum() / mcls_pos.sum() * mcls_pos + mcls_neg 
            mcls_lss = nn.functional.binary_cross_entropy(mcls_pred, mcls_pos, reduction='none')
            mcls_lss = (mcls_weights * mcls_lss).mean()

            fcls_weights = fcls_neg.sum() / fcls_pos.sum() * fcls_pos + fcls_neg 
            fcls_lss = nn.functional.binary_cross_entropy(fcls_pred, fcls_pos, reduction='none')
            fcls_lss = (fcls_weights * fcls_lss).mean()            
                
            cls_lss = mcls_lss + fcls_lss    
            cls_batch_lss.append(cls_lss)
            
            # Plot cls metric
            plot_cls_metric(mcls_pred, mcls_pos, cthres, train_vis.plots.cls_mid)
            plot_cls_metric(fcls_pred, fcls_pos, cthres, train_vis.plots.cls_fine)
            
            # Plot statis
            train_vis.plots.cls_ratios.mpos_gt.append(mcls_pos.sum().item() / N)
            train_vis.plots.cls_ratios.fpos_gt.append(fcls_pos.sum().item() / N)
            train_vis.plots.loss.cls_mid.append(mcls_lss.item())
            train_vis.plots.loss.cls_fine.append(fcls_lss.item())
                        
            # Epipolar loss
            mids_gt = torch.where(cdist < epi_dthres[0], ones, zeros).nonzero(as_tuple=False).flatten()
            fids_gt = torch.where(mdist < epi_dthres[1], ones, zeros).nonzero(as_tuple=False).flatten()
            #lprint_(f'{len(mdist)} {len(mids_gt)} {len(fdist)} {len(fids_gt)}')
            
            if len(fids_gt) == 0 and len(mids_gt) == 0: 
                skipped += 1
                continue

            epi_mid = mdist[mids_gt].mean() if len(mids_gt) > 0 else torch.tensor(0).to(mdist)
            epi_fine = fdist[fids_gt].mean() if len(fids_gt) > 0 else torch.tensor(0).to(fdist)
            epi_lss = emid_weight * epi_mid + efine_weight * epi_fine
            epi_batch_lss.append(epi_lss)                        

            # Plot epi dists
            if len(mids_gt) > 0:  
                train_vis.plots.loss.epi_mid.append(epi_mid.item())
                train_vis.plots.match_dist.mmid_gt.append(epi_mid.item())
                train_vis.plots.match_dist.cmid_gt.append(cdist[mids_gt].mean().item())                
                        
            if len(fids_gt) > 0:
                train_vis.plots.loss.epi_fine.append(epi_fine.item())                
                train_vis.plots.match_dist.ffid_gt.append(epi_fine.item())
                train_vis.plots.match_dist.mfid_gt.append(mdist[fids_gt].mean().item())

        # Total loss
        cls_loss = torch.stack(cls_batch_lss).mean() if len(cls_batch_lss) > 0 else torch.tensor(0.0, requires_grad=True).to(net.device)
        epi_loss = torch.stack(epi_batch_lss).mean() if len(epi_batch_lss) > 0 else torch.tensor(0.0, requires_grad=True).to(net.device)        
        loss = cls_loss_weight * cls_loss + epi_loss
        train_vis.plots.loss.pair.append(loss.item())

        # Optimize
        net.optim_step_(loss)

        # Monitor memory usage    
        rss, vms = get_sys_mem()
        train_vis.plots.mem.rss.append(rss)
        train_vis.plots.mem.vms.append(vms)
        
        gpu_maloc, gpu_mres = get_gpu_mem()
        train_vis.plots.mem.gpu_maloc.append(gpu_maloc)
        train_vis.plots.mem.gpu_mres.append(gpu_mres)
        torch.cuda.empty_cache()
        
        # Update plots periocially
        if i % plot_step == 0 and i > 0:                        
            train_vis.plot(epoch=epoch + (i / len(train_loader)))
            lprint_('Batch:{} Loss:{}'.format(i, train_vis.get_plot_print(train_vis.plots.loss)))
            lprint_('Cls_mid:{}'.format(train_vis.get_plot_print(train_vis.plots.cls_mid)))
            lprint_('Cls_fine:{}'.format(train_vis.get_plot_print(train_vis.plots.cls_fine)))            
            lprint_('Match:{}\n'.format(train_vis.get_plot_print(train_vis.plots.match_dist)))
    
    # Always update plots in the end of an epoch            
    train_vis.plot(epoch=epoch + 1)                
    lprint_('>Epoch:{} Skipped:{} Loss:{}'.format(epoch + 1, skipped, train_vis.get_plot_print(train_vis.plots.loss)))
    lprint_('Cls_mid:{}'.format(train_vis.get_plot_print(train_vis.plots.cls_mid)))
    lprint_('Cls_fine:{}'.format(train_vis.get_plot_print(train_vis.plots.cls_fine)))            
    lprint_('Match:{}'.format(train_vis.get_plot_print(train_vis.plots.match_dist)))
    
def main():
    np.set_printoptions(precision=3)   
    args = parse_agrs()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    make_deterministic(args.seed)
    print(args)
    
    # Init data loader
    match_npy_name = args.match_npy 
    match_npy = os.path.join(args.pair_root, match_npy_name)    
    pair_type = match_npy_name.replace('megadepth_pairs.', '').replace('_imrat1.5','').replace('.npy','')
    data_tag = 'Mega.' + pair_type
    train_set = ImMatchDatasetMega(args.data_root, match_npy, wt=480, ht=320)
    train_loader = data.DataLoader(train_set, batch_size=args.batch, shuffle=True)
    
    # Init output dir names
    if args.prefix is not '':
        odir_tag = args.prefix + '.' + data_tag
    else:
        odir_tag = data_tag
    odir_tag += '.freeze{}'.format(args.freeze_feat)
    if args.change_stride:
        odir_tag += '.cs'    
    if args.pretrain:
        odir_tag += '.pretrain'
    
    # fe1234nc0.9ep50-100_cls{}lr5e-4..
    feat_tag = 'ks{}fe{}'.format(args.ksize, ''.join([str(v) for v in args.feat_idx]))
    thres_tag = 'ep{}-{}cls{}-{}'.format(args.epi_dthres[0], args.epi_dthres[1],
                                              args.cls_dthres[0], args.cls_dthres[1])
    train_tag = '_wcls{}wepi{}-{}.lr{}'.format(args.weight_cls, args.weight_epi[0], 
                                               args.weight_epi[1], args.lr_init)
    
    if args.weight_decay > 0:
        train_tag += 'wd{}'.format(args.weight_decay)
    if args.lr_decay:
        decay_type = args.lr_decay[0]
        if decay_type == 'step':
            train_tag = '{}lrst{}-{}'.format(train_tag, args.lr_decay[1], args.lr_decay[2])
        elif decay_type == 'multistep':
            train_tag = '{}lrms{}-{}'.format(train_tag, args.lr_decay[1], args.lr_decay[2])

    exp_tag = '{}{}{}'.format(feat_tag, thres_tag, train_tag)   
    
    # Regressor
    regress_tag = '{}{}_conv{}dim{}str{}fc{}_psz{}-{}a{}'.format(args.feat_comb, args.ptmax,
                                                                ''.join(map(str, args.conv_kers)),
                                                                '-'.join(map(str, args.conv_dims)), 
                                                                '-'.join(map(str, args.conv_strs)),
                                                                '-'.join(map(str, args.fc_dims)),
                                                                args.psize[0], args.psize[1], 
                                                                args.panc)        
    if args.shared:
        regress_tag += '.shared'
        
    # Create output dirs and log file
    out_dir = os.path.join(args.out_dir, odir_tag, exp_tag, regress_tag)
    args.out_dir = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log = open(os.path.join(out_dir, 'log.txt'), 'a')
    lprint_ = lambda ms: lprint(ms, log)
    lprint_(config2str(args))
    lprint_('Log dir {}'.format(out_dir))
    lprint_(f'>>>Load dataset:{data_tag}, train:{len(train_loader.dataset)}')
            
    
    # Initialize visdom
    env = '{}.{}_{}{}_{}'.format(odir_tag, feat_tag, thres_tag, train_tag, regress_tag)
    server = args.visdom_host
    port = args.visdom_port
    lprint_('>>Visdom server: {} port: {} env: {}'.format(server, port, env))    
    train_vis = get_visdom_plots(prefix='train', env=env, server=server, port=port)
    test_vis = get_visdom_plots(prefix='test', env=env, server=server, port=port)
    
    # Initialize model
    config, best_vals = init_model_config(args, lprint_)
    config.freeze_nc = True                
    net = Patch2Pix(config)
    if args.weight_epi[0] == 0:
        lprint_('Freeze regressor_mid ...')        
        for param in net.regress_mid.parameters():
            param.requires_grad = False
            
    lprint_('Params backboone={} ncn={} regress_mid={} regress_fine={}'.format(
        count_parameters(net.extract),
        count_parameters(net.ncn),
        count_parameters(net.regress_mid), 
        count_parameters(net.regress_fine)        
    ))
    lprint_('Set geo dist: sampson distance')            
    net.geo_dist_fn = sampson_dist

    
    # Training and validation
    t0 = time.time()    
    lprint_('Start training from {} to {} ..'.format(config.start_epoch, args.epochs))  
    for epoch in range(config.start_epoch, args.epochs):        
                
        # Always train on normally matching pairs
        lprint_('\n>>>Epoch {} training...'.format(epoch+1))        
        lprint_('>>>Current_lr={}\n'.format(net.optimizer.param_groups[0]['lr']))
        t1 = time.time()        
        train_epoch(epoch, net, train_loader, train_vis, args, lprint_)
        lprint_('Epoch training time: {:.2f}s'.format(time.time() - t1))        
        
        # Validation 
        net.panc = 1  # Hard set topk to 1   
        lprint_(f'Validation setting: panc={net.panc}')
        
        # Always save last ckpt 
        save_ckpt(net, epoch, out_dir, best_vals=best_vals, last_ckpt=True)    

        # Save model periodically
        if (epoch + 1) % args.save_step == 0:
            save_ckpt(net, epoch, out_dir, best_vals=best_vals)                                       
            

        # Eval immatch 
        try:
            res = eval_immatch_val_sets(net, 
                                        data_root=f'{args.data_root}/immatch_benchmark/val_dense',
                                        ksize=2, imsize=1024,
                                        eval_type='fine', io_thres=0.5, 
                                        sample_max=150, lprint_=lprint_)

            # Save the best model based on immatch
            qt_err, pass_rate = res    
            rate = 0.34 * pass_rate[0] + 0.33 * pass_rate[4] + 0.33 * pass_rate[9]  # % < 1/5/10 px
            if qt_err < best_vals[2] or rate > best_vals[3]:
                if qt_err < best_vals[2]:
                    best_vals[2] = qt_err
                if rate > best_vals[3]:
                    best_vals[3] = rate
                save_ckpt(net, epoch, out_dir, best_vals=best_vals, name='immatch_best_ckpt')
                lprint_('>>Save best immatch model: epoch={} qt={:.3f} rate={:.2f}%'.format(epoch+1, qt_err, rate))
        
        except:
            lprint_('Failed to eval immatch')
            res = None
                    
        # Update the learning rate    
        if net.lr_scheduler:           
            net.lr_scheduler.step()
    lprint_('Finished, time:{:.4f}s'.format(time.time() - t0))
    log.close()
    
    
if __name__ == '__main__':
    main()
        