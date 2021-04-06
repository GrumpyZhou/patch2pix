import os
from PIL import Image
import numpy as np
import torch.utils.data as data

from utils.datasets.preprocess import get_tuple_transform_ops
from utils.eval.measure import sampson_distance
from utils.eval.geometry import pose2fund

class ImMatchDatasetMega(data.Dataset):
    '''Data wrapper for train image-matching with triplets'''
    def __init__(self, data_root, match_file, scene_list=None, wt=480, ht=320, item_type='triplet'):
        print('\nInitialize ImMatchDatasetMega...')        
        self.dataset = 'MegaDepth_undistort'
        self.data_root = os.path.join(data_root, self.dataset)
        self.match_file = match_file
        self.transform_ops = get_tuple_transform_ops(resize=(ht, wt), normalize=True)
        self.wt, self.ht = wt, ht
        self.item_type = item_type
        
        # Initialize data
        self.ims = {}            # {scene: {im: imsize}}
        self.pos_pair_pool = []  # [pair]
        self.load_pairs(scene_list)
        self.Fs = {}
        self.Ks = {}
        
        
    def load_im(self, im_ref, crop=None):
        im = Image.open(im_ref)
        if crop:
            dw, dh = crop  
            im = np.array(im)

            # Crop from right and buttom to keep the target aspect ratio
            h, w, _ = im.shape
            im = im[0: h - int(dh), 0: w - int(dw)]
            #print(h, w, im.shape)
            im = Image.fromarray(im)
        return im
            
    def load_pairs(self, scene_list=None):        
        match_dict = np.load(self.match_file, allow_pickle=True).item()
        self.scenes = scene_list if scene_list else match_dict.keys()
        print('Loading data from {}'.format(self.match_file))                
        
        num_ims = 0
        for sc in self.scenes:
            self.pos_pair_pool += match_dict[sc]['pairs']
            self.ims[sc] = match_dict[sc]['ims']
            num_ims += len(match_dict[sc]['ims'])    
        print('Loaded scenes {} ims: {} pos pairs:{}'.format(len(self.scenes), num_ims, len(self.pos_pair_pool)))            
    
    def get_fundmat(self, pair, im1, im2):        
        def scale_intrinsic(K, wi, hi):
            sx, sy = self.wt / wi, self.ht /  hi
            sK = np.array([[sx, 0, 0],
                           [0, sy, 0],
                           [0, 0, 1]])
            return sK.dot(K)
        
        pair_key = (pair.im1, pair.im2)                
        if pair_key not in self.Fs:        
            # Recompute camera intrinsic matrix due to the resize
            K1 = scale_intrinsic(pair.K1, im1.width, im1.height)
            K2 = scale_intrinsic(pair.K2, im2.width, im2.height)

            # Calculate F
            F = pose2fund(K1, K2, pair.R, pair.t)        
            self.Fs[pair_key] = (F, K1, K2)

            # Sanity check
            # scale = np.array([[im1.width/self.wt, im1.height/self.ht, im2.width/self.wt, im2.height/self.ht]])
            # matches = pair.sanity_matches * scale        
            # dists = sampson_distance(matches[:, :2], matches[:,2:], F)
            # print(np.mean(dists))
        return self.Fs[pair_key]
    
    def __getitem__(self, index):
        """
        Batch dict:
            - 'src_im': anchor image
            - 'pos_im': positive image sample to the anchor
            - 'neg_im': negative image sample to the anchor
            - 'im_pair_refs': path of images (src, pos, neg)
            - 'pair_data': namedtuple contains relative pose information between src and pos ims.
        """
        data_dict = {}
        
        # Load positive pair data
        pair = self.pos_pair_pool[index]
        im_src_ref = os.path.join(self.data_root, pair.im1)
        im_pos_ref = os.path.join(self.data_root, pair.im2)
        im_src = self.load_im(im_src_ref, crop=pair.crop1)
        im_pos = self.load_im(im_pos_ref, crop=pair.crop2)
        
        # Select a negative image from other scences        
        if self.item_type == 'triplet':
            other_scenes = list(self.scenes)
            other_scenes.remove(pair.im1.split('/')[0])
            neg_scene = np.random.choice(other_scenes)
            im_neg_data = np.random.choice(self.ims[neg_scene])
            im_neg_ref = os.path.join(self.data_root, im_neg_data.name)
            im_neg = self.load_im(im_neg_ref, crop=im_neg_data.crop)
            im_neg = self.transform_ops([im_neg])[0]  
            #print(im_neg.shape)
        else:
            im_neg = None
            im_neg_ref = None
            
        
        # Compute fundamental matrix before RESIZE
        F, K1, K2 = self.get_fundmat(pair, im_src, im_pos) 
        
        # Process images
        im_src, im_pos = self.transform_ops((im_src, im_pos))
        #print(im_src.shape, im_pos.shape)
        
        # Wrap data item
        data_dict = {'src_im': im_src, 
                     'pos_im': im_pos, 
                     'neg_im': im_neg, 
                     'im_pair_refs': (im_src_ref, im_pos_ref, im_neg_ref),
                     'F': F,
                     'K1': K1,
                     'K2': K2
                     }

        return data_dict
    
    def __len__(self):
        return len(self.pos_pair_pool)
    
    def __repr__(self):
        fmt_str = 'ImMatchDatasetMega scenes:{} data type:{}\n'.format(len(self.scenes), self.item_type)
        fmt_str += 'Number of data pairs: {}\n'.format(self.__len__())
        fmt_str += 'Image root location: {}\n'.format(self.data_root)
        fmt_str += 'Match file: {}\n'.format(self.match_file)
        fmt_str += 'Transforms: {}\n'.format(self.transform_ops.__repr__().replace('\n', '\n    '))
        return fmt_str  

