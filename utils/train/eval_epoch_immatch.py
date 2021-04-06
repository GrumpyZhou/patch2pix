from argparse import Namespace
import os
import numpy as np
import time
from utils.colmap.data_loading import load_model_ims 
from utils.eval.geometry import abs2relapose, pose2fund
from transforms3d.quaternions import quat2mat
from utils.eval.model_helper import estimate_matches
from utils.eval.measure import check_inliers_distr, eval_matches_relapose, sampson_distance


def eval_immatch_val_sets(net, data_root='data/immatch_benchmark/val_dense', 
                          ksize=2, eval_type='fine', io_thres=0.5, 
                          ncn_thres=0.0, imsize=1024,
                          rthres=0.5, sample_max=300, min_overlap=0.3, 
                          lprint_=print):
    net.eval()
    np.random.seed(0)  # Deterministic pairs for evaluation
    scenes = os.listdir(data_root)
    lprint_(f'\n>>Eval on immatch: rthres={rthres} eval_type={eval_type} ov<{min_overlap} '
            f'nc={ncn_thres} ksize={ksize} io={io_thres} im={imsize}')            

    errs = Namespace(qt=[], fdist=[], cdist=[], indist=[], irat=[], 
                     num_matches=[], num_inls=[], match_failed=[], geo_failed=[])
    count = 0
    start_time = time.time()
    for scene in scenes:    
        # Load scene ims and pairs
        model_dir = os.path.join(data_root, scene, 'dense/sparse')
        im_dir = os.path.join(data_root, scene, 'dense/images')
        ims = load_model_ims(model_dir)
        ov_pair_dict = np.load(os.path.join(model_dir,'ov_pairs.npy'), allow_pickle=True).item()
        pair_names = ov_pair_dict[min_overlap]
        total = len(pair_names)
        if total > sample_max:
            np.random.shuffle(pair_names)
            pair_names = pair_names[0:sample_max]

        for i, (im1_name, im2_name) in enumerate(pair_names):
            im1 = ims[im1_name]
            im2 = ims[im2_name]
            t, q = abs2relapose(im1.c, im2.c, im1.q, im2.q)
            R = quat2mat(q)
            F = pose2fund(im1.K, im2.K, R, t)

            # Compute matches
            im1_path = os.path.join(im_dir, im1_name)
            im2_path = os.path.join(im_dir, im2_name)  
            count += 1
            try:
                matches, scores, coarse_matches = estimate_matches(net, im1_path, im2_path, 
                                                                   ksize=ksize,
                                                                   ncn_thres=ncn_thres, 
                                                                   eval_type=eval_type,
                                                                   io_thres=io_thres,
                                                                   imsize=imsize)
            except:
                errs.match_failed.append((im1_path, im2_path))
                continue
            
            N = len(matches)
            cdists = sampson_distance(coarse_matches[:, 0:2], coarse_matches[:, 2:4], F)
            fdists = sampson_distance(matches[:, 0:2], matches[:, 2:4], F)
            errs.cdist.append(cdists)
            errs.fdist.append(fdists)
            errs.num_matches.append(N)            

            try:
                # Eval relaposes
                terr, qerr, inls = eval_matches_relapose(matches, im1.K, im2.K, q, t, rthres)
                indists = fdists[inls]
                irat = len(inls) / N
                # print(f't={terr:.2f} q={qerr:.2f} inls={len(inls)}')
            except:
                errs.geo_failed.append((im1_path, im2_path))
                continue
            errs.qt.append(max(terr, qerr))
            errs.irat.append(irat)
            errs.indist.append(indists)
            errs.num_inls.append(len(inls))        
    runtime = time.time() - start_time
    lprint_(f'Pairs {count} match_failed={len(errs.match_failed)} geo_failed={len(errs.geo_failed)} '
            f'num_matches={np.mean(errs.num_matches):.2f} irat={ np.mean(errs.irat):.3f} time:{runtime:.2f}s')
    
    bins = [0, 1e-2, 1, 5, 10, 25, 50, 100, 2500, 1e5]
    cdist_print = check_inliers_distr(errs.cdist, bins=bins, tag='cdist')
    fdist_ratios, fdist_print = check_inliers_distr(errs.fdist, bins=bins, tag='fdist', return_ratios=True)
    indist_ratios, indist_print = check_inliers_distr(errs.indist, bins=bins, tag='indist', return_ratios=True) 
    lprint_(cdist_print)        
    lprint_(fdist_print)
    lprint_(indist_print)
    
    pass_rate = np.array([100.0*np.mean(np.array(errs.qt) < thre) for thre in range(1, 11, 1)])    
    qt_err_mean = np.mean(errs.qt)
    qt_err_med = np.median(errs.qt)    
    lprint_('Pose err: qt_mean={:.2f}/{:.2f} qt<[1-10]deg:{}'.format(qt_err_mean, qt_err_med, pass_rate))
    
    return qt_err_mean, pass_rate
    