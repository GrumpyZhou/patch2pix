import torch
import random
from torchvision import transforms
import numpy as np
from PIL import Image

def load_im_tensor(im_path, device, imsize=None, with_gray=True):    
    im = Image.open(im_path)

    # Resize
    wt, ht = wo, ho = im.width, im.height
    if imsize and max(wo, ho) > imsize and imsize > 0:
        scale = imsize / max(wo, ho)
        ht, wt = int(round(ho * scale)), int(round(wo * scale))
        im = im.resize((wt, ht), Image.BICUBIC)    
    scale = (wo / wt, ho / ht)
    
    # Gray
    gray = None
    if with_gray:
        im_gray = np.array(im.convert('L'))
        gray = transforms.functional.to_tensor(im_gray).unsqueeze(0).to(device)
        
    # RGB  
    im = transforms.functional.to_tensor(im)
    im = transforms.functional.normalize(im , mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im = im.unsqueeze(0).to(device)
    if with_gray:
        return im, gray, scale
    return im, scale

def load_im_flexible(im_path, k_size=2, upsample=16, imsize=None, crop_square=False):
    """Load image for matching on fly.
    The image can be arbitrary size and will be processed to 
    fulfil the format required for matching.    
    """    
    # Load original image
    img = Image.open(im_path)
    img = img.convert('RGB')
    wo, ho = img.width, img.height
    if not (imsize and imsize > 0):
        imsize = max(wo, ho)
    elif imsize > max(wo, ho):     # Disable upsampling
        imsize = max(wo, ho)
        
    wt, ht = cal_rescale_size(image_size=imsize, w=wo, h=ho, k_size=k_size, 
                              scale_factor=1./upsample, no_print=True)    

    # Resize image and transform to tensor
    ops = get_tuple_transform_ops(resize=None, normalize=True)
    img = transforms.functional.resize(img, (ht, wt), Image.BICUBIC)
    img = ops([img])[0]
    
    # Mainly for beauty plotting
    if crop_square:    
        _, h, w = img.shape
        img = img[:, :w,:]
        
    scale = (wo / wt, ho / ht)    
    return img, scale

def crop_from_bottom_right(w, h, target_ratio=1.5, min_ratio=1.3, max_ratio=1.7):
    ratio = w / h    
    if ratio < min_ratio or ratio > max_ratio:
        return None
    if ratio == target_ratio:
        return 0, 0
    if ratio > target_ratio:
        # Cut the width
        dh = h % 2
        ht = h - dh
        dw = w - ht * target_ratio
        wt = w - dw

    if ratio < target_ratio:
        # Cut the height
        dw = w % 3
        wt = w - dw    
        dh = h - wt / target_ratio
        ht = h - dh
    return dw, dh

def cal_rescale_size(image_size, w, h, k_size=2, scale_factor=1/16, no_print=False):
    # Calculate target image size (lager side=image_size)
    wt = int(np.floor(w/(max(w, h)/image_size)*scale_factor/k_size)/scale_factor*k_size)
    ht = int(np.floor(h/(max(w, h)/image_size)*scale_factor/k_size)/scale_factor*k_size)
    N = wt * ht * scale_factor * scale_factor / (k_size ** 2)
    if not no_print:
        print('Target size {} Original: (w={},h={}), Rescaled: (w={},h={}) , matches resolution: {}'.format(image_size, w, h, 
                                                                                                        wt, ht, N))
    return wt, ht

def get_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize))
    if normalize: 
        ops.append(TupleToTensorScaled())
        ops.append(TupleNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))  # Imagenet mean/std
    else:
        if unscale:
            ops.append(TupleToTensorUnscaled())
        else:
            ops.append(TupleToTensorScaled())
    return TupleCompose(ops)

class ToTensorScaled(object):
    '''Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]'''
    def __call__(self, im):
        im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
        im /= 255.0 
        return torch.from_numpy(im)

    def __repr__(self):
        return 'ToTensorScaled(./255)'
    
class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()
        
    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return 'TupleToTensorScaled(./255)'
    
class ToTensorUnscaled(object):
    '''Convert a RGB PIL Image to a CHW ordered Tensor'''
    def __call__(self, im):    
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return 'ToTensorUnscaled()'

class TupleToTensorUnscaled(object):
    '''Convert a RGB PIL Image to a CHW ordered Tensor'''
    def __init__(self):
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return 'TupleToTensorUnscaled()'
    
class TupleResize(object):
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(size, Image.BICUBIC)

    def __call__(self, im_tuple):
        return [self.resize(im) for im in im_tuple]
    
    def __repr__(self):
        return 'TupleResize(size={})'.format(self.size)
    
class TupleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        return [self.normalize(im) for im in im_tuple]

    def __repr__(self):
        return 'TupleNormalize(mean={}, std={})'.format(self.mean, self.std)

class TupleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple):
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string    