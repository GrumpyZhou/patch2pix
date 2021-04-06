import os
from argparse import Namespace
import numpy as np
from utils.colmap.read_database import COLMAPDataLoader
from utils.colmap.read_write_model import *

def sav_model_multi_ov_pairs(model_dir, overlaps):
    sav_file_path = os.path.join(model_dir, 'ov_pairs.npy')
    if os.path.exists(sav_file_path):
        ov_pair_dict = np.load(sav_file_path, allow_pickle=True).item()
        all_exists = True        
        for k in overlaps:
            if k not in ov_pair_dict:
                all_exists = False
        if all_exists:
            print('All overlaps have been computed.')
            return ov_pair_dict

    ov_pair_dict = {}        
    images = read_images_binary(os.path.join(model_dir, 'images.bin'))
    im_ids = list(images.keys())
    overlap_scores, _ = cal_overlap_scores(im_ids, images)               

    for min_overlap in overlaps:
        if min_overlap in ov_pair_dict:
            print(f'ov>{min_overlap} exists, skip.')
            continue
        valid_scores = np.logical_and(overlap_scores >= min_overlap, overlap_scores < 1)
        pair_ids = np.vstack(np.where(valid_scores)).T
        pair_names = []
        for id1, id2 in pair_ids:
            im1 = images[im_ids[id1]].name
            im2 = images[im_ids[id2]].name
            pair_names.append((max(im1, im2), min(im1, im2)))
        print(f'ov>{min_overlap} pairs: {len(pair_names)}')
        ov_pair_dict[min_overlap] = pair_names        
    np.save(sav_file_path, ov_pair_dict)
    return ov_pair_dict    
    
def load_model_ov_pairs(model_dir, min_overlap=0.3):    
    images = read_images_binary(os.path.join(model_dir, 'images.bin'))
    im_ids = list(images.keys())
    overlap_scores, _ = cal_overlap_scores(im_ids, images)               
    valid_scores = np.logical_and(overlap_scores >= min_overlap, overlap_scores < 1)
    pair_ids = np.vstack(np.where(valid_scores)).T
    pair_names = []
    for id1, id2 in pair_ids:
        im1 = images[im_ids[id1]].name
        im2 = images[im_ids[id2]].name
        pair_names.append((max(im1, im2), min(im1, im2)))
    print('Loaded ov>{} pairs: {}'.format(min_overlap, len(pair_names)))
    return pair_names

def cal_overlap_scores(im_ids, images):
    N = len(im_ids)
    overlap_scores = np.eye(N)
    im_3ds = [np.where(images[i].point3D_ids > 0)[0] for i in im_ids]

    for i in range(N):
        im1 = images[im_ids[i]]
        pts1 = im_3ds[i]    
        for j in range(N):
            if j <= i :
                continue
            im2 = images[im_ids[j]]
            pts2 = im_3ds[j]        
            ov = len(np.intersect1d(pts1, pts2)) / max(len(pts1), len(pts2))
            overlap_scores[i, j] = ov
    nums_3d = np.array([len(v) for v in im_3ds])
    return overlap_scores, nums_3d

def load_model_ims(model_dir):
    cameras = read_cameras_binary(os.path.join(model_dir, 'cameras.bin'))
    images = read_images_binary(os.path.join(model_dir, 'images.bin'))
    #print(len(cameras), len(images))   
    imdict = {}
    for i in images:
        cid = images[i].camera_id
        if cid not in cameras:
            continue        
        data = parse_data(images[i], cameras[cid])
        imdict[data.name] = data
    return imdict

def cam_params_to_matrix(params, model):
    if model == 'SIMPLE_PINHOLE':
        f, ox, oy = params
        K = np.array([[f, 0, ox], [0, f, oy], [0, 0, 1]])
    elif model == 'PINHOLE':
        f1, f2, ox, oy = params
        K = np.array([[f1, 0, ox], [0, f2, oy], [0, 0, 1]])
    elif model == 'SIMPLE_RADIAL':
        f, ox, oy, _ = params
        K = np.array([[f, 0, ox], [0, f, oy], [0, 0, 1]])
    elif model == 'RADIAL':
        f, ox, oy, _, _ = params
        K = np.array([[f, 0, ox], [0, f, oy], [0, 0, 1]])
    return K

def parse_data(im, cam):
    # Extract information from Image&Camera objects
    K = cam_params_to_matrix(cam.params, cam.model)
    q = im.qvec
    R = qvec2rotmat(q)
    t = im.tvec
    c = - R.T.dot(t)
    return Namespace(name=im.name, K=K, c=c, q=q, R=R, id=im.id)

def load_colmap_matches(db_path, pair_names):
    # Loading data from colmap database
    db_loader = COLMAPDataLoader(db_path)
    keypoints = db_loader.load_keypoints(key_len=6)
    images = db_loader.load_images(name_based=True)   
    pair_ids = [(images[im1][0], images[im2][0]) for im1, im2 in pair_names]
    db_matches = db_loader.load_pair_matches(pair_ids)
    match_dict = {}
    for pname, pid in zip(pair_names, pair_ids):
        (im1, im2) = pid
        kpts1 = keypoints[im1]
        kpts2 = keypoints[im2]
        if pid not in db_matches:
            matches = None
        else:
            key_ids = db_matches[pid]
            N = key_ids.shape[0]
            matches = np.zeros((N, 4))
            for j in range(N):
                k1, k2 = key_ids[j,:]
                x1, y1 = kpts1[k1][0:2]
                x2, y2 = kpts2[k2][0:2]
                matches[j, :] = [x1, y1, x2, y2]    
        match_dict[pname] = matches
    return match_dict

def export_intrinsics_txt(model_dir, sav_path): 
    cameras = read_cameras_binary(os.path.join(model_dir, 'cameras.bin'))
    images = read_images_binary(os.path.join(model_dir, 'images.bin'))
    with open(sav_path, 'w') as f:
        for imid in images:
            im = images[imid]
            cid = im.camera_id
            if cid not in cameras:
                continue
            cam = cameras[cid]
            model = cam.model
            w, h = cam.width, cam.height
            params = cam.params
            
            # Line format: im SIMPLE_RADIAL 1600 1200 1199.91 800 600 -0.0324314
            line = f'{im.name} {model} {w} {h} '
            for p in params:
                line += f'{p} '
            line += '\n'
            f.write(line)
            f.flush()
    print('Finished, save to ', sav_path)
        
def parse_camera_matrices(intrinsic_txt):
    with open(intrinsic_txt) as f:
        # Line format: im SIMPLE_RADIAL 1600 1200 1199.91 800 600 -0.0324314
        intrinsic_lines = f.readlines()
        camera_matrices = {}
        for line in intrinsic_lines:
            cur = line.split()
            name, model = cur[0:2]
            params = [float(v) for v in cur[4::]]
            K = cam_params_to_matrix(params, model)
            camera_matrices[name] = K
    return camera_matrices    
