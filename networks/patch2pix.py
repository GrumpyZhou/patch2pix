import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import networks.resnet as resnet
from networks.modules import *
from networks.utils import select_local_patch_feats, filter_coarse
from networks.ncn.model import MutualMatching, NeighConsensus  
from networks.ncn.extract_ncmatches import corr_to_matches

class Patch2Pix(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.backbone = config.backbone
        self.change_stride = config.change_stride        
        self.upsample = 16
        self.feats_downsample = [1, 2, 2, 2, 2]
        feat_dims = [3, 64, 64, 128, 256]  # Resnet34 block feature out dims 
        
        # Initialize necessary network components
        self.extract = resnet.__dict__[self.backbone]()
        if self.change_stride:
            self.extract.change_stride(target='layer3')
            self.upsample //= 2            
            self.feats_downsample[-1] = 1        
        print(f'Initialize Patch2Pix: backbone={self.backbone} '
              f'cstride={self.change_stride} upsample={self.upsample}')
        
        self.combine = FeatCorrelation(shape='4D')
        self.ncn = NeighConsensus(kernel_sizes=[3, 3], channels=[16, 1])
        
        # Initialize regressor
        self.regressor_config = config.regressor_config
        if not self.regressor_config:
            # If no regressor defined, model only computes coarse matches
            self.regress_mid = None
            self.regress_fine = None
        else:
            print(f'Init regressor {self.regressor_config}')
            self.regr_batch = config.regr_batch
            self.feat_idx = config.feat_idx            
            feat_dim = 0  # Regressor's input feature dim
            for idx in self.feat_idx:
                feat_dim += feat_dims[idx]           
            self.regressor_config.feat_dim = feat_dim            
            self.ptype = ['center', 'center']
            self.psize = config.regressor_config.psize            
            self.pshift = config.regressor_config.pshift
            self.panc = config.regressor_config.panc
            self.shared = config.regressor_config.shared
            self.regress_mid = FeatRegressNet(self.regressor_config, psize=self.psize[0])
            if self.shared:
                self.regress_fine = self.regress_mid
                self.psize[1] = self.psize[0]
            else:
                self.regress_fine = FeatRegressNet(self.regressor_config, psize=self.psize[1])
            
        self.to(self.device)
        self.init_weights_(weights_dict=config.weights_dict, pretrained=True)        
           
        if config.training:
            self.freeze_feat = config.freeze_feat
            # Freeze (part of) the backbone
            print('Freezing feature extractor params upto layer {}'.format(self.freeze_feat))
            for i, param in enumerate(self.extract.parameters()):         
                # Resnet34 layer3=[48:87] blocks:
                # 0=[48:57] 1=[57:63] 2=[63:69]
                # 3=[69:75] 4=[75:81] 5=[81:87] 
                if i < self.freeze_feat:
                    param.requires_grad = False

                # Always freeze resnet layer4, since never used
                if i >= 87:
                    param.requires_grad = False        
            
            config.optim_config.start_epoch = config.start_epoch
            self.set_optimizer_(config.optim_config)
    
    def set_optimizer_(self, optim_config): 
        params = []
        if self.regress_mid:
            params += list(self.regress_mid.parameters()) 
        if self.regress_fine and not self.shared:
            params += list(self.regress_fine.parameters())         
        params += list(self.ncn.parameters())
        if self.freeze_feat < 87:
            params += list(self.extract.parameters())[self.freeze_feat:87]
        self.optimizer, self.lr_scheduler = init_optimizer(params, optim_config)
        print('Init optimizer, items: {}'.format(len(params)))

    def optim_step_(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()            
                
    def init_weights_(self, weights_dict=None, pretrained=True):
        print('Xavier initialize all model parameters')
        self.apply(xavier_init_func_)
        if pretrained:
            self.extract.load_pretrained_()
        if weights_dict:
            if len(weights_dict.items()) == len(self.state_dict()):
                print('Reload all model parameters from weights dict')
                self.load_state_dict(weights_dict)
            else:
                print('Reload part of model parameters from weights dict')
                self.load_state_dict(weights_dict, strict=False)       
                
    def load_batch_(self, batch, dtype='pair'):
        im_src = batch['src_im'].to(self.device)
        im_pos = batch['pos_im'].to(self.device)
        Fs = batch['F'].to(self.device)            
        if dtype == 'triplet':
            im_neg = batch['neg_im'].to(self.device)
            return im_src, im_pos, im_neg, Fs
        return im_src, im_pos, Fs    

    def forward_coarse_match(self, feat1, feat2, ksize=1):
        # Feature normalization
        feat1 = L2Normalize(feat1, dim=1)
        feat2 = L2Normalize(feat2, dim=1)
        
        # Feature correlation
        corr4d = self.combine(feat1, feat2)
        
        # Do 4d maxpooling for relocalization
        delta4d = None
        if ksize > 1:
            corr4d, max_i, max_j, max_k, max_l = maxpool4d(corr4d, k_size=ksize)
            delta4d = (max_i,max_j,max_k,max_l)
        corr4d = MutualMatching(corr4d)            
        corr4d = self.ncn(corr4d)
        corr4d = MutualMatching(corr4d)        
        return corr4d, delta4d

    def parse_regressor_out(self, out, psize, ptype, imatches, max_val):
        w1, h1, w2, h2 = max_val
        offset = out[:, :4] # N, 4
        offset = psize * torch.tanh(nn.functional.relu(offset))        
        if ptype == 'center':
            shift = psize // 2
            offset -= shift
        fmatches = imatches.float() + offset
        io_probs = out[:, 4]
        io_probs = torch.sigmoid(io_probs)
        
        # Prevent out of range
        x1 = fmatches[:, 0].clamp(min=0, max=w1)
        y1 = fmatches[:, 1].clamp(min=0, max=h1)
        x2 = fmatches[:, 2].clamp(min=0, max=w2)
        y2 = fmatches[:, 3].clamp(min=0, max=h2)
        fmatches = torch.stack([x1, y1, x2, y2], dim=-1)
        return fmatches, io_probs

    def forward_fine_match_mini_batch(self, feats1, feats2, ibatch, imatches,
                                      psize, ptype, regressor):
        # ibatch: to index the feature map
        # imatches: input coarse matches, N, 4
        N = imatches.shape[0]
        _, _, h1, w1 = feats1[0].shape
        _, _, h2, w2 = feats2[0].shape
        max_val = [w1, h1, w2, h2]
        
        f1s, f2s, _, _ = select_local_patch_feats(feats1, feats2,
                                                  ibatch, imatches,
                                                  feat_idx=self.feat_idx,
                                                  feats_downsample=self.feats_downsample,
                                                  psize=psize,
                                                  ptype=ptype)    
        # Feature normalization
        f1s = L2Normalize(f1s, dim=0)  # D, N*psize*psize
        f2s = L2Normalize(f2s, dim=0)  # D, N*psize*psize

        # Reshaping: -> (D, N, psize, psize) -> (N, D, psize, psize)
        f1s = f1s.view(-1, N, psize, psize).permute(1, 0, 2, 3)
        f2s = f2s.view(-1, N, psize, psize).permute(1, 0, 2, 3)   

     
        # From im1 to im2       
        out = regressor(f1s, f2s)  # N, 5
        fmatches, io_probs = self.parse_regressor_out(out, psize, ptype, imatches, max_val)
        return fmatches, io_probs
    
    def forward_fine_match(self, feats1, feats2, coarse_matches, 
                           psize, ptype, regressor):
        batch_size = self.regr_batch
        masks = []
        fine_matches = []
        for ibatch, imatches in enumerate(coarse_matches):
            # Use mini-batch if too many matches
            N = imatches.shape[0]
            if N > batch_size:                
                batch_inds = [batch_size*i for i in range(N // batch_size + 1)]
                if batch_inds[-1] < N:
                    if N -  batch_inds[-1] == 1:
                        # Special case, slicing leads to 1-dim missing
                        batch_inds[-1] = N
                    else:
                        batch_inds += [N]
                fmatches = []
                io_probs = []
                for bi, (ist, ied) in enumerate(zip(batch_inds[0:-1], batch_inds[1::])):
                    mini_results = self.forward_fine_match_mini_batch(feats1, feats2, 
                                                                      ibatch, imatches[ist:ied],
                                                                      psize, ptype, regressor)
                    fmatches.append(mini_results[0])
                    io_probs.append(mini_results[1])
                fmatches = torch.cat(fmatches, dim=0).squeeze()
                io_probs = torch.cat(io_probs, dim=0).squeeze()
            else:
                fmatches, io_probs = self.forward_fine_match_mini_batch(feats1, feats2, 
                                                                        ibatch, imatches,
                                                                        psize, ptype, regressor)
            fine_matches.append(fmatches)
            masks.append(io_probs)                        
        return fine_matches, masks

    def forward(self, im1, im2, ksize=1, return_feats=False):
        if return_feats:
            feat1s=[]
            feat2s=[]
            self.extract.forward_all(im1, feat1s, early_feat=True)        
            self.extract.forward_all(im2, feat2s, early_feat=True)
            feat1 = feat1s[-1]
            feat2 = feat2s[-1]
        else:
            feat1 = self.extract(im1, early_feat=True)        
            feat2 = self.extract(im2, early_feat=True)     # Shared weights
        
        corr4d, delta4d = self.forward_coarse_match(feat1, feat2, ksize=ksize)
        
        if return_feats:
            return corr4d, delta4d, feat1s, feat2s                            
        else:
            return corr4d, delta4d
        
        
    def predict_coarse(self, im1, im2, ksize=2, ncn_thres=0.0,
                       mutual=False, center=True):
        corr4d, delta4d = self.forward(im1, im2, ksize)
        coarse_matches, match_scores = self.cal_coarse_matches(corr4d, delta4d, ksize=ksize, 
                                                               upsample=self.upsample, center=center)
        
        # Filter coarse matches
        coarse_matches, match_scores = filter_coarse(coarse_matches, match_scores, ncn_thres, mutual)
        return coarse_matches, match_scores
    
    def predict_fine(self, im1, im2, ksize=2, ncn_thres=0.0,
                     mutual=True, return_all=False):
        corr4d, delta4d, feats1, feats2 = self.forward(im1, im2, ksize=ksize, return_feats=True)
        coarse_matches, match_scores = self.cal_coarse_matches(corr4d, delta4d, ksize=ksize,
                                                               upsample=self.upsample, center=True)
        # Filter coarse matches
        coarse_matches, match_scores = filter_coarse(coarse_matches, match_scores, ncn_thres, mutual)
        
        # Locate initial anchors
        coarse_matches = self.shift_to_anchors(coarse_matches)
        
        # Mid level matching
        mid_matches, mid_scores = self.forward_fine_match(feats1, feats2, 
                                                          coarse_matches, 
                                                          psize=self.psize[0],
                                                          ptype=self.ptype[0],
                                                          regressor=self.regress_mid)
        
        # Fine level matching
        fine_matches, fine_scores = self.forward_fine_match(feats1, feats2, 
                                                            mid_matches,
                                                            psize=self.psize[1],
                                                            ptype=self.ptype[1],
                                                            regressor=self.regress_fine)
        if return_all:
            return fine_matches, fine_scores, mid_matches, mid_scores, coarse_matches 
        return fine_matches, fine_scores, coarse_matches  
    
    def refine_matches(self, im1, im2, coarse_matches, io_thres):
        # Handle empty coarse matches
        if len(coarse_matches) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0, 4))

        if type(coarse_matches) == np.ndarray:
            coarse_matches_ = torch.from_numpy(coarse_matches).to(self.device).unsqueeze(0)  # 1, N, 4            
        elif type(coarse_matches) == torch.Tensor:
            coarse_matches_ = coarse_matches.unsqueeze(0)  # 1, N, 4
            coarse_matches = coarse_matches.cpu().data.numpy()
        
        # Extract local features
        feat1s=[]
        feat2s=[]
        self.extract.forward_all(im1, feat1s, early_feat=True)        
        self.extract.forward_all(im2, feat2s, early_feat=True)    

        # Mid level matching
        mid_matches, mid_scores = self.forward_fine_match(feat1s, feat2s, 
                                                          coarse_matches_, 
                                                          psize=self.psize[0],
                                                          ptype=self.ptype[0],
                                                          regressor=self.regress_mid)

        # Fine level matching
        fine_matches, fine_scores = self.forward_fine_match(feat1s, feat2s, 
                                                            mid_matches,
                                                            psize=self.psize[1],
                                                            ptype=self.ptype[1],
                                                            regressor=self.regress_fine)
        refined_matches = fine_matches[0].cpu().data.numpy()
        scores = fine_scores[0].cpu().data.numpy()
        
        # Further filtering with threshold
        if io_thres > 0:
            pos_ids = np.where(scores > io_thres)[0]
            if len(pos_ids) > 0:
                coarse_matches = coarse_matches[pos_ids]
                refined_matches = refined_matches[pos_ids]
                scores = scores[pos_ids]
        return refined_matches, scores, coarse_matches
    
    def cal_coarse_score(self, corr4d, normalize='softmax'):
        if normalize is None:
            normalize = lambda x: x
        elif normalize == 'softmax':     
            normalize = lambda x: nn.functional.softmax(x, 1)
        elif normalize == 'l1':
            normalize = lambda x: x / (torch.sum(x, dim=1, keepdim=True) + 0.0001)
        
        # Mutual matching score
        batch_size, _, h1, w1, h2, w2 = corr4d.shape
        nc_B_Avec=corr4d.view(batch_size, h1*w1, h2, w2)
        nc_A_Bvec=corr4d.view(batch_size, h1, w1, h2*w2).permute(0,3,1,2) # 
        nc_B_Avec = normalize(nc_B_Avec)
        nc_A_Bvec = normalize(nc_A_Bvec)
        scores_B,_= torch.max(nc_B_Avec, dim=1)
        scores_A,_= torch.max(nc_A_Bvec, dim=1)
        scores_AB = torch.cat([scores_A.view(-1, h1*w1), scores_B.view(-1, h2*w2)], dim=1)
        score = scores_AB.mean()
        return score
    
    def cal_coarse_matches(self, corr4d, delta4d, ksize=1, do_softmax=True,
                           upsample=16, sort=False, center=True, pshift=0):
        
        # Original nc implementation: only max locations
        (xA_, yA_, xB_, yB_, score_) = corr_to_matches(corr4d, delta4d=delta4d,
                                                       do_softmax=do_softmax,
                                                       ksize=ksize)
        (xA2_, yA2_, xB2_, yB2_, score2_) = corr_to_matches(corr4d, delta4d=delta4d,
                                                            do_softmax=do_softmax,
                                                            ksize=ksize,
                                                            invert_matching_direction=True)
        xA_ = torch.cat((xA_, xA2_), 1)
        yA_ = torch.cat((yA_, yA2_), 1)
        xB_ = torch.cat((xB_, xB2_), 1)
        yB_ = torch.cat((yB_, yB2_), 1)
        score_ = torch.cat((score_, score2_),1)
        
        # Sort as descend
        if sort:
            sorted_index = torch.sort(-score_)[1]
            xA_ = torch.gather(xA_, 1, sorted_index)  # B, 1, N
            yA_ = torch.gather(yA_, 1, sorted_index)
            xB_ = torch.gather(xB_, 1, sorted_index)
            yB_ = torch.gather(yB_, 1, sorted_index)
            score_ = torch.gather(score_, 1, sorted_index)  # B, N

        xA_ = xA_.unsqueeze(1)
        yA_ = yA_.unsqueeze(1)
        xB_ = xB_.unsqueeze(1)
        yB_ = yB_.unsqueeze(1)        
        # Create matches and upscale to input resolution
        matches_ = upsample * torch.cat([xA_, yA_, xB_, yB_], dim=1).permute(0, 2, 1) # B, N, 4
        if center:
            delta = upsample // 2
            matches_ += torch.tensor([[delta, delta, delta, delta]]).unsqueeze(0).to(matches_)                        
        return matches_, score_
    
    def shift_to_anchors(self, matches): 
        pshift = self.pshift
        panc = self.panc
        if panc == 1:
            return matches
        
        # Move pt1/pt2 to its upper-left, upper-right, down-left, down-right
        # location by pshift, leading to 4 corner anchors
        # Then take center vs corner from two directions as new matches 
        shift_template = torch.tensor([
            [-pshift, -pshift, 0, 0],
            [pshift, -pshift,  0, 0],
            [-pshift, pshift,  0, 0],
            [pshift, pshift,   0, 0],
            [0, 0, -pshift, -pshift],                
            [0, 0, pshift, -pshift],
            [0, 0, -pshift, pshift],
            [0, 0, pshift, pshift]
        ]).to(self.device)

        matches_ = []
        for imatches in matches:
            imatches =  imatches.unsqueeze(1) + shift_template # N, 16, 4
            imatches = imatches.reshape(-1, 4)
            matches_.append(imatches)
        return matches_
    