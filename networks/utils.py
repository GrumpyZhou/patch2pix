import torch
import numpy as np

def select_local_patch_feats(feats1, feats2, ibatch, imatches, 
                             feat_idx=[1, 2, 3, 4],
                             feats_downsample=[1, 2, 2, 2, 2],
                             psize=16, ptype='center'):
    dy, dx = torch.meshgrid(torch.arange(psize), torch.arange(psize))
    dx = dx.flatten().view(1, -1).to(imatches.device)
    dy = dy.flatten().view(1, -1).to(imatches.device)

    if ptype == 'center':
        shift = psize // 2
        dy -= shift
        dx -= shift    
    
    _, _, h1, w1 = feats1[0].shape
    _, _, h2, w2 = feats2[0].shape
    x1, y1, x2, y2 = imatches.permute(1, 0).long()

    # Ids for local patch
    get_x_pids = lambda x, w, ds: ((x.view(-1, 1) + dx).view(-1) // ds).long().clamp(min=0, max=w//ds-1)
    get_y_pids = lambda y, h, ds: ((y.view(-1, 1) + dy).view(-1) // ds).long().clamp(min=0, max=h//ds-1)

    # Collect features for local matches
    f1s, f2s = [], []
    for j, (fmap1, fmap2) in enumerate(zip(feats1, feats2)):
        if j not in feat_idx:
            continue 
        ds = np.prod(feats_downsample[0:j+1])
        f1s.append(fmap1[ibatch, :, get_y_pids(y1, h1, ds), get_x_pids(x1, w1, ds)])
        f2s.append(fmap2[ibatch, :, get_y_pids(y2, h2, ds), get_x_pids(x2, w2, ds)])
        
    f1s = torch.cat(f1s, dim=0) # D, N*16
    f2s = torch.cat(f2s, dim=0) # D, N*16
    return f1s, f2s, dx.squeeze(), dy.squeeze()

def filter_coarse(coarse_matches, match_scores, ncn_thres=0.0, mutual=True, ptmax=None):
    matches = []
    scores = []
    for imatches, iscores in  zip(coarse_matches, match_scores):
        _, ids, counts = np.unique(imatches.cpu().data.numpy(), axis=0, return_index=True, return_counts=True)
        if mutual:
            # Consider only if they are multual consistant 
            ids = ids[counts > 1]
            #print(len(imatches), len(ids))
            
        if len(ids) > 0:
            iscores = iscores[ids]
            imatches = imatches[ids]

        # NC score filtering
        ids = torch.nonzero(iscores.flatten() > ncn_thres, as_tuple=False).flatten()
        
        # Cut or fill upto ptmax for memory control
        if ptmax: 
            if len(ids) == 0:
                # insert a random match
                ids = torch.tensor([0, 0, 0, 0]).long()
            iids = np.arange(len(ids))
            np.random.shuffle(iids)
            iids = np.tile(iids, (ptmax // len(ids) + 1))[:ptmax]
            ids = ids[iids]
            
        if len(ids) > 0: 
            iscores = iscores[ids]
            imatches = imatches[ids]
            
        matches.append(imatches)
        scores.append(iscores)
        
    return matches, scores

def sym_epi_dist(matches, F, sqrt=True, eps=1e-8):
    # matches: Nx4
    # F: 3x3
    N = matches.shape[0]
    matches = matches.to(F)
    ones = torch.ones((N,1)).to(F)
    p1 = torch.cat([matches[:, 0:2] , ones], dim=1) 
    p2 = torch.cat([matches[:, 2:4] , ones], dim=1)

    # l2=F*x1, l1=F^T*x2
    l2 = F.matmul(p1.transpose(1, 0)) # 3,N
    l1 = F.transpose(1, 0).matmul(p2.transpose(1, 0))
    dd = (l2.transpose(1, 0) * p2).sum(dim=1)

    sqrt = False
    if sqrt:
        d = dd.abs() * (1.0 / (eps + l1[0, :] ** 2 + l1[1, :] ** 2).sqrt() + 1.0 / (eps + l2[0, :] ** 2 + l2[1, :] ** 2).sqrt())    
    else:
        d = dd ** 2 * (1.0 / (eps + l1[0, :] ** 2 + l1[1, :] ** 2) + 1.0 / (eps + l2[0, :] ** 2 + l2[1, :] ** 2))
    return d.float()

def sampson_dist(matches, F, eps=1e-8):
    # First-order approximation to reprojection error
    # matches: Nx4
    # F: 3x3
    N = matches.shape[0]
    matches = matches.to(F)
    ones = torch.ones((N,1)).to(F)
    p1 = torch.cat([matches[:, 0:2] , ones], dim=1) 
    p2 = torch.cat([matches[:, 2:4] , ones], dim=1)

    # l2=F*x1, l1=F^T*x2
    l2 = F.matmul(p1.transpose(1, 0)) # 3,N
    l1 = F.transpose(1, 0).matmul(p2.transpose(1, 0))
    dd = (l2.transpose(1, 0) * p2).sum(dim=1)
    d = dd ** 2 / (eps + l1[0, :] ** 2 + l1[1, :] ** 2 + l2[0, :] ** 2 + l2[1, :] ** 2)   
    return d.float()

