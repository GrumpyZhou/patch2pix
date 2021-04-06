import numpy as np
import torch
from PIL import Image

def plot_imlist_to_pdf(ims, sav_name, figsize=(50, 35), dpi=250):
    import matplotlib.pyplot as plt
    row_num = len(plot_ims)
    fig = plt.figure(figsize=figsize)
    for i in range(row_num):    
        ax = fig.add_subplot(row_num, 1, i+1)    
        ax.imshow(plot_ims[i])
        ax.axis('off')    
    fig.tight_layout()
    fig.savefig(sav_name, dpi=dpi,  bbox_inches='tight')      
    plt.show()
    
def plot_imlist(imlist):
    '''Plot a list of images in a row'''
    import matplotlib.pyplot as plt
    if type(imlist) is str:
        fig = plt.figure(figsize=(5, 3))
        imlist = [imlist]
    else:
        fig = plt.figure(figsize=(25, 3))
    num = len(imlist)
    for i, im in enumerate(imlist):
        im = Image.open(im)
        ax = fig.add_subplot(1, num, i+1)
        ax.imshow(im)
    plt.show()
    
def plot_pair(pair):
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(20, 5))            
    im1 = Image.open(pair[0])
    im2 = Image.open(pair[1])
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(im1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(im2)
    plt.show()
    
def plot_triple(pair):
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(20, 5))            
    im1 = Image.open(pair[0])
    im2 = Image.open(pair[1])
    im3 = Image.open(pair[2])    
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(im1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(im2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(im3)
    plt.show()    
    
def torch2rgb(im):
    im = im.squeeze().permute(1, 2, 0)
    if im.device.type == 'cuda':
        im = im.data.cpu().numpy()
    else:
        im = im.data.numpy()
    return im.astype(np.uint8)

def undo_normalize_scale(im):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    im = im * std + mean
    im *= 255.0
    return im.astype(np.uint8)

def recover_im_from_torch(torch_im):
    im = torch_im
    im = im.squeeze().permute(1, 2, 0).cpu().data.numpy()    
    im = undo_normalize_scale(im)
    return im

def scatter_pts(im, pts, unnormalize=True):
    import matplotlib.pyplot as plt

    if isinstance(im, torch.Tensor):            
        im = im.squeeze().permute(1, 2, 0).cpu().data.numpy()
        if unnormalize:
            im = undo_normalize_scale(im)            
        I = Image.fromarray(im)
    elif isinstance(im, np.ndarray):
        I = Image.fromarray(im)
    elif isinstance(im, str):
        I = Image.open(im)
    else:
        I = im
    plt.imshow(I)
    ax = plt.gca()
    for x in pts:
        ax.add_artist(plt.Circle((x[0], x[1]), radius=1, color='red'))
    plt.gcf().set_dpi(150)
    plt.show()
    
def plot_pair_loader(data_loader, row_max=2, normalize_and_scale=False):
    import matplotlib.pyplot as plt
    for i, batch in enumerate(data_loader):
        print('>>>>>>>>>')
        fig1 = plt.figure(figsize=(20, 5))
        fig2 = plt.figure(figsize=(20, 5))
        num = len(batch['im_pairs'][0])
        for j in range(num):
            im_pair = batch['im_pairs']
            im1 = im_pair[0][j,:, :, :].permute(1, 2, 0).data.numpy()
            im2 = im_pair[1][j,:, :, :].permute(1, 2, 0).data.numpy()
            if normalize_and_scale:
                im1 = undo_normalize_scale(im1)
                im2 = undo_normalize_scale(im2)
            else:
                im1 = im1.astype(np.uint8)
                im2 = im2.astype(np.uint8)
            ax1 = fig1.add_subplot(1, num, j+1)
            ax1.imshow(im1)
            ax2 = fig2.add_subplot(1, num, j+1)
            ax2.imshow(im2)
        plt.show()
        if i >= row_max:
            break
            
def plot_immatch_loader(data_loader, normalize_and_scale=False, num_sample=2, 
                        axis='on', dtype='triplet'):
    import matplotlib.pyplot as plt
    
    num = data_loader.batch_size
    count = 0   
    if dtype == 'pair':
        ncols = 2
    else:
        ncols = 3
        
    for i, batch in enumerate(data_loader):
        print('Batch >>>>>>>>>')
        for j in range(num):
            fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 5))
            im_src = batch['src_im'][j, :, :, :].permute(1, 2, 0).data.numpy()
            im_pos = batch['pos_im'][j, :, :, :].permute(1, 2, 0).data.numpy()

            im_src = undo_normalize_scale(im_src) if normalize_and_scale else im_src.astype(np.uint8)
            im_pos = undo_normalize_scale(im_pos) if normalize_and_scale else im_pos.astype(np.uint8)
            axs[0].imshow(im_src)
            axs[0].axis(axis)
            axs[1].imshow(im_pos)
            axs[1].axis(axis)
       
            if ncols == 3:
                im_neg = batch['neg_im'][j, :, :, :].permute(1, 2, 0).data.numpy()
                im_neg = undo_normalize_scale(im_neg) if normalize_and_scale else im_neg.astype(np.uint8)
                axs[2].imshow(im_neg)
                axs[2].axis(axis)            
            count += 1
            plt.gcf().set_dpi(350)                        
            plt.show()
            
        if count > num_sample:
            break
            
def plot_triple_loader(data_loader, normalize_and_scale=False, num_sample=2, 
                       axis='on', dtype='triplet'):
    import matplotlib.pyplot as plt
    
    num = data_loader.batch_size
    count = 0   
    if dtype == 'pair':
        ncols = 3
    else:
        ncols = 4
        
    for i, batch in enumerate(data_loader):
        print('Batch >>>>>>>>>')
        for j in range(num):
            fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 5))
            im1 = batch['im1'][j, :, :, :].permute(1, 2, 0).data.numpy()
            im2 = batch['im2'][j, :, :, :].permute(1, 2, 0).data.numpy()
            im3 = batch['im3'][j, :, :, :].permute(1, 2, 0).data.numpy()

            im1 = undo_normalize_scale(im1) if normalize_and_scale else im_src.astype(np.uint8)
            im2 = undo_normalize_scale(im2) if normalize_and_scale else im_pos.astype(np.uint8)
            im3 = undo_normalize_scale(im3) if normalize_and_scale else im_pos.astype(np.uint8)

            axs[0].imshow(im1)
            axs[0].axis(axis)
            axs[1].imshow(im2)
            axs[1].axis(axis)
            axs[2].imshow(im3)
            axs[2].axis(axis)
       
            if ncols == 4:
                im_neg = batch['neg_im'][j, :, :, :].permute(1, 2, 0).data.numpy()
                im_neg = undo_normalize_scale(im_neg) if normalize_and_scale else im_neg.astype(np.uint8)
                axs[3].imshow(im_neg)
                axs[3].axis(axis)            
            count += 1
            plt.gcf().set_dpi(350)                        
            plt.show()
            
        if count > num_sample:
            break            
            
def plot_matches_cv(im1, im2, matches, inliers=None, Npts=1000, radius=3, dpi=350, sav_fig=None, ret_im=False):
    import matplotlib.pyplot as plt
    import cv2

    # Read images and resize        
    if isinstance(im1, torch.Tensor):            
        im1 = im1.squeeze().permute(1, 2, 0).cpu().data.numpy()
        im2 = im2.squeeze().permute(1, 2, 0).cpu().data.numpy()
        I1 = undo_normalize_scale(im1)
        I2 = undo_normalize_scale(im2)           
    elif isinstance(im1, str):
        I1 = np.array(Image.open(im1))
        I2 = np.array(Image.open(im2))
    else:
        I1 = im1
        I2 = im2
    
    if inliers is None:
        inliers = np.arange(len(matches))
        
    if Npts < len(inliers):
        inliers = inliers[:Npts]        
    
    # Only matches
    p1s = []
    p2s = []
    dmatches = []
    for i, (x1, y1, x2, y2) in enumerate(matches):
        if i in inliers:    
            p1s.append(cv2.KeyPoint(x1, y1, 1))
            p2s.append(cv2.KeyPoint(x2, y2, 1))
            j = len(p1s) - 1
            dmatches.append(cv2.DMatch(j, j, 1))
    print('Plot {} matches'.format(len(dmatches)))

    I3 = cv2.drawMatches(I1, p1s, I2, p2s, dmatches, None)

    fig = plt.figure(figsize=(50, 50))
    axis = fig.add_subplot(1, 1, 1)
    axis.imshow(I3)
    axis.axis('off') 
    if sav_fig:        
        fig.savefig(sav_fig, dpi=150,  bbox_inches='tight')      
    plt.show()    
    if ret_im:
        return I3
            
def plot_matches(im1, im2, matches, inliers=None, Npts=None, lines=False,
                 unnormalize=True, radius=5, dpi=150, sav_fig=False,
                 colors=None):
    import matplotlib.pyplot as plt

    # Read images and resize        
    if isinstance(im1, torch.Tensor):            
        im1 = im1.squeeze().permute(1, 2, 0).cpu().data.numpy()
        im2 = im2.squeeze().permute(1, 2, 0).cpu().data.numpy()
    
        if unnormalize:
            im1 = undo_normalize_scale(im1)
            im2 = undo_normalize_scale(im2)
        else:
            im1 = im1.astype(np.uint8)
            im2 = im2.astype(np.uint8)            
        I1 = Image.fromarray(im1)
        I2 = Image.fromarray(im2)    
    elif isinstance(im1, np.ndarray):
        I1 = Image.fromarray(im1)
        I2 = Image.fromarray(im2)    
    elif isinstance(im1, str):
        I1 = Image.open(im1)
        I2 = Image.open(im2)
    else:
        I1 = im1
        I2 = im2
        
    w1, h1 = I1.size
    w2, h2 = I2.size 

    if h1 <= h2:
        scale1 = 1;
        scale2 = h1/h2
        w2 = int(scale2 * w2)
        I2 = I2.resize((w2, h1))
    else:
        scale1 = h2/h1
        scale2 = 1
        w1 = int(scale1 * w1)
        I1 = I1.resize((w1, h2))
    catI = np.concatenate([np.array(I1), np.array(I2)], axis=1)

    # Load all matches
    match_num = matches.shape[0]
    if inliers is None:
        if Npts is not None:
            Npts = Npts if Npts < match_num else match_num
        else:
            Npts = matches.shape[0]
        inliers = range(Npts) # Everthing as an inlier
    else:
        if Npts is not None and Npts < len(inliers):
            inliers = inliers[:Npts]
    print('Plotting inliers: ', len(inliers))

    x1 = scale1*matches[inliers, 0]
    y1 = scale1*matches[inliers, 1]
    x2 = scale2*matches[inliers, 2] + w1
    y2 = scale2*matches[inliers, 3]
    c = np.random.rand(len(inliers), 3) 
    
    if colors is not None:
        c = colors
    
    # Plot images and matches
    fig = plt.figure(figsize=(30, 20))
    axis = plt.gca()#fig.add_subplot(1, 1, 1)
    axis.imshow(catI)
    axis.axis('off')
    
    #plt.imshow(catI)
    #ax = plt.gca()
    for i, inid in enumerate(inliers):
        # Plot
        axis.add_artist(plt.Circle((x1[i], y1[i]), radius=radius, color=c[i,:]))
        axis.add_artist(plt.Circle((x2[i], y2[i]), radius=radius, color=c[i,:]))
        if lines:
            axis.plot([x1[i], x2[i]], [y1[i], y2[i]], c=c[i,:], linestyle='-', linewidth=radius)
    if sav_fig:        
        fig.savefig(sav_fig, dpi=dpi,  bbox_inches='tight')      
    plt.show()    

    
def plot_epilines(im1, im2, x1s, x2s, F, Npts=50, 
                  figsize=(30, 20), unnormalize=True, dpi=350):
    """
    Args:
        - x1s, x2s: shape (N, 3)

    """
    import matplotlib.pyplot as plt
    
     # Read images and resize
    if isinstance(im1, torch.Tensor):            
        im1 = im1.squeeze().permute(1, 2, 0).cpu().data.numpy()
        im2 = im2.squeeze().permute(1, 2, 0).cpu().data.numpy()
    
        if unnormalize:
            im1 = undo_normalize_scale(im1)
            im2 = undo_normalize_scale(im2)
            
        I1 = Image.fromarray(im1)
        I2 = Image.fromarray(im2)  
    elif isinstance(im1, np.ndarray):
        I1 = Image.fromarray(im1)
        I2 = Image.fromarray(im2)    
    elif isinstance(im1, str):
        I1 = Image.open(im1)
        I2 = Image.open(im2)
    else:
        I1 = im1
        I2 = im2
        
    w1, h1 = I1.size
    w2, h2 = I2.size 
    
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    ax1.imshow(I1)
    ax2 = fig.add_subplot(212)
    ax2.imshow(I2)

    num_pts = x1s.shape[0]
    Npts = min(num_pts, Npts)
    ids = np.random.permutation(num_pts)[0:Npts]
    colors = np.random.rand(Npts, 3) 

    for p1, p2, color in zip(x1s[ids, :], x2s[ids, :], colors):
        ax1.add_artist(plt.Circle((p1[0], p1[1]), radius=3, color=color))
        ax2.add_artist(plt.Circle((p2[0], p2[1]), radius=3, color=color))

        # Calculate epilines
        l2 = F.dot(p1)   # On the second image
        a, b, c = l2
        ax2.plot([0, w2 ], [-c/b, -(c + a*w2)/b], c=color, linestyle='-', linewidth=0.4)

        l1 = F.T.dot(p2)
        a, b, c = l1
        ax1.plot([0, w1 ], [-c/b, -(c + a*w1)/b], c=color, linestyle='-', linewidth=0.4)
    plt.gcf().set_dpi(dpi)
    plt.show()    
