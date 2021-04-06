import numpy as np

get_statis = lambda arr: 'Size={} Min={:.2f} Max={:.2f} Mean={:.2f} Median={:.2f}'.format(
                                arr.shape, np.min(arr), np.max(arr), np.mean(arr), np.median(arr))

def expand_homo_ones(arr2d, axis=1):
    """Raise 2D array to homogenous coordinates
    Args:
        - arr2d: (N, 2) or (2, N)
        - axis: the axis to append the ones
    """    
    if axis == 0:
        ones = np.ones((1, arr2d.shape[1]))
    else:
        ones = np.ones((arr2d.shape[0], 1))      
    return np.concatenate([arr2d, ones], axis=axis)

def sampson_distance(pts1, pts2, F, homos=True, eps=1e-8):
    """Calculate symmetric epipolar distance between 2 sets of points
    Args:
        - pts1, pts2: points correspondences in the two images, 
          each has shape of (num_points, 2)
        - F: fundamental matrix that fulfills x2^T*F*x1=0, 
          where x1 and x2 are the correspondence points in the 1st and 2nd image 
    Return:
        A vector of (num_points,), containing root-squared epipolar distances
          
    """
    
    # Homogenous coordinates
    if homos:
        pts1 = expand_homo_ones(pts1, axis=1) #if pts1.shape[1] == 2 else pts1
        pts2 = expand_homo_ones(pts2, axis=1) #if pts2.shape[1] == 2 else pts2
    
    # l2=F*x1, l1=F^T*x2
    l2 = np.dot(F, pts1.T) # 3,N
    l1 = np.dot(F.T, pts2.T)
    dd = np.sum(l2.T * pts2, 1)  # Distance from pts2 to l2   
    d = dd ** 2 / (eps + l1[0, :] ** 2 + l1[1, :] ** 2 + l2[0, :] ** 2 + l2[1, :] ** 2)   
    return d


def symmetric_epipolar_distance(pts1, pts2, F, homos=True, sqrt=False):
    """Calculate symmetric epipolar distance between 2 sets of points
    Args:
        - pts1, pts2: points correspondences in the two images, 
          each has shape of (num_points, 2)
        - F: fundamental matrix that fulfills x2^T*F*x1=0, 
          where x1 and x2 are the correspondence points in the 1st and 2nd image 
    Return:
        A vector of (num_points,), containing root-squared epipolar distances
          
    """
    
    # Homogenous coordinates
    if homos:
        pts1 = expand_homo_ones(pts1, axis=1) #if pts1.shape[1] == 2 else pts1
        pts2 = expand_homo_ones(pts2, axis=1) #if pts2.shape[1] == 2 else pts2
    
    # l2=F*x1, l1=F^T*x2
    l2 = np.dot(F, pts1.T) # 3,N
    l1 = np.dot(F.T, pts2.T)
    dd = np.sum(l2.T * pts2, 1)  # Distance from pts2 to l2
    
    if sqrt:
        # The one following DFM and find correspondence paper
        d = np.abs(dd) * (1.0 / np.sqrt(l1[0, :] ** 2 + l1[1, :] ** 2) + 1.0 / np.sqrt(l2[0, :] ** 2 + l2[1, :] ** 2))
    else: 
        # Original one as in MVG Hartley.
        d = dd ** 2 * (1.0 / (l1[0, :] ** 2 + l1[1, :] ** 2) + 1.0 /(l2[0, :] ** 2 + l2[1, :] ** 2))
    return d

def cal_vec_angle_error(label, pred, eps=1e-14):
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)

    v1 = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + eps)
    v2 = label / (np.linalg.norm(label, axis=1, keepdims=True) + eps)
    d = np.sum(np.multiply(v1,v2), axis=1, keepdims=True) 
    d = np.clip(d, a_min=-1, a_max=1)
    error = np.degrees(np.arccos(d))    
    return error.squeeze()

def cal_quat_angle_error(label, pred, eps=1e-14):
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    q1 = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + eps)
    q2 = label / (np.linalg.norm(label, axis=1, keepdims=True) + eps)
    d = np.abs(np.sum(np.multiply(q1,q2), axis=1, keepdims=True))
    d = np.clip(d, a_min=-1, a_max=1)
    error = 2 * np.degrees(np.arccos(d))
    return error.squeeze()

def cal_rot_angle_error(Rgt, Rpred):
    # Identical to quaternion angular error
    return np.rad2deg(np.arccos((np.trace(Rpred.T.dot(Rgt)) - 1) / 2))

def eval_matches_relapose(matches, K1, K2, q_, t_, cv_thres=1.0):
    from utils.eval.geometry import  matches2relapose_cv
    from transforms3d.quaternions import mat2quat
    
    p1 = matches[:,:2]
    p2 = matches[:,2:4]        
    E, inls, R, t = matches2relapose_cv(p1, p2, K1, K2, rthres=cv_thres)
    
    # Calculate relative angle errors
    terr = cal_vec_angle_error(t.squeeze(), t_)
    qerr = cal_quat_angle_error(mat2quat(R), q_)
    return terr, qerr, inls

def check_inliers_distr(inlier_dists, 
                        bins = [0, 1e-2, 1, 5, 10, 25, 50, 100, 400, 2500, 1e5],
                        tag='', return_ratios=False):
    if not inlier_dists:
        if return_ratios:
            return None, ''
        return ''
    inlier_ratios = []
    Npts = []
    for dists in inlier_dists:
        N = len(dists)
        if N == 0:
            continue
        Npts.append(N)
        hists = np.histogram(dists, bins)[0]
        inlier_ratios.append(hists / N)

    ratio_print = '{} Sample:{} N(mean/max/min):{:.0f}/{:.0f}/{:.0f}\nRatios(%):'.format(tag, len(inlier_dists), np.mean(Npts), 
                                                                                         np.max(Npts), np.min(Npts))

    ratios = []
    for val, low, high in zip(np.mean(inlier_ratios, axis=0), bins[0:-1], bins[1::]):
        ratio_print = '{} [{},{})={:.2f}'.format(ratio_print, low, high, 100*val)
        ratios.append(100*val)
    if return_ratios:
            return ratios, ratio_print
    return ratio_print

def check_data_hist(data_list, bins, tag='', return_hist=False):
    if not data_list:
        return ''
    hists = []
    means = []
    for data in data_list:
        if len(data) == 0:
            continue
        nums = np.histogram(data, bins)[0]
        hists.append(nums / len(data))
        means.append(np.mean(data))

    hist_print = f'{tag} mean={np.mean(means):.2f}'
    mean_hists = np.mean(hists, axis=0)
    for val, low, high in zip(mean_hists, bins[0:-1], bins[1::]):
        hist_print += ' [{},{})={:.2f}'.format(low, high, 100 * val)
    if return_hist:
        return mean_hists, hist_print
    return hist_print
