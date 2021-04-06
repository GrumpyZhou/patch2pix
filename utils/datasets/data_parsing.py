import os
import numpy as np

def parse_3d_points_from_nvm(nvm_file):
    """
    Formats of nvm file:
        <Number of cameras>   <List of cameras>
        <Number of 3D points> <List of points>
        <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
        <Measurement> = <Image index> <Feature Index> <xy>
    """
    cams = []       # List image frames 
    cam_points = {} # Map key: index of frame, value: list of indices of 3d points that are visible to this frame.
    points = []     # List of 3d points in the reconstruction model
    
    print('Read 3D points from {}'.format(nvm_file))
    with open(nvm_file, 'r') as f:
        next(f)    # Skip headding lines
        next(f)
        
        # Load images
        cam_num = int(next(f).split()[0])
        for i in range(cam_num):
            line = next(f)
            frame = line.split()[0]
            cams.append(frame)
            cam_points[frame] = []
            
        next(f)  # Skip the separation line
        point_num = int(next(f).split()[0])
        for i in range(point_num):
            line = next(f)
            cur = line.split()
            X = cur[0:3]
            points.append(X)
            measure_num = int(cur[6])
            for j in range(measure_num):
                idx = int(cur[7+j*4])
                frame = cams[idx]
                cam_points[frame].append(i)
    print('Loading finished: camera frames {}, total 3d points {}'.format(len(cam_points), len(points)))
    return (points, cam_points)

def parse_abs_pose_txt(fpath):
    """Absolute pose label format: 
        3 header lines
        list of samples with format: 
            image x y z w p q r
    """
    
    pose_dict = {}
    f = open(fpath)
    for line in f.readlines()[3::]:    # Skip 3 header lines
        cur = line.split(' ')
        c = np.array([float(v) for v in cur[1:4]], dtype=np.float32)
        q = np.array([float(v) for v in cur[4:8]], dtype=np.float32)
        im = cur[0]
        pose_dict[im] = (c, q)
    f.close()
    return pose_dict

class CambridgeIntrinsics:
    scenes = ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']
    def __init__(self, base_dir, scene, wt=1920, ht=1080, w=1920, h=1080):
        assert scene in self.scenes
        self.base_dir = base_dir
        self.scene = scene
        self.wt, self.ht = wt, ht
        self.w, self.h = w, h
        self.ox, self.oy = w / 2, h / 2
        self.sK = np.array([[wt / w, 0, 0],
                            [0, ht / h, 0],
                            [0, 0, 1]])
        self.focals = self.get_focals()
        self.im_list = list(self.focals.keys())
        self.intrinsic_matrices = {}        
        for im in self.im_list:
            f = self.focals[im]
            K = np.array([[f, 0, self.ox], 
                          [0, f, self.oy], 
                          [0, 0, 1]], dtype=np.float32)
            K = self.sK.dot(K)
            self.intrinsic_matrices[im] = K

    def get_focals(self):
        focals = {}
        nvm = os.path.join(self.base_dir, self.scene,'reconstruction.nvm')
        with open(nvm, 'r') as f:
            # Skip headding lines
            next(f)   
            next(f)
            cam_num = int(f.readline().split()[0])
            print('Loading focals scene: {} cameras: {}'.format(self.scene, cam_num))
            
            focals = {}
            for i in range(cam_num):
                line = f.readline()
                cur = line.split()
                focals[cur[0].replace('jpg', 'png')] = float(line.split()[1])
        return focals
        
    def get_intrinsic_matrices(self):
        return self.intrinsic_matrices
        
    def get_im_intrinsics(self, im):
        return self.intrinsic_matrices[im]
    
def get_positive_pairs(cam_points, imlist, thres_min=0.15, thres_max=0.8):    
    """
    Args:
        cam_points: (key:cam, val: [3d_point_ids])
        thres_min, thres_max: min/max thresholds for overlapped 3D points 
    Return:
        pairs: {(im1, im2): PosPair})
    """
    from argparse import Namespace
    from transforms3d.quaternions import quat2mat
    from utils.eval.geometry import abs2relapose

    # Pairwise overlapping calculation
    pairs = []
    overlaps = []
    total_num_pos = 0
    for i, im1 in enumerate(imlist):
        for j, im2 in enumerate(imlist):
            if j <= i:
                continue
        
            # Calculate overlapping
            p1 = cam_points[im1.name.replace('png', 'jpg')]
            p2 = cam_points[im2.name.replace('png', 'jpg')]
            p12 = list(set(p1).intersection(p2))  # Common visible points
            score = min(1.0 * len(p12) / len(p1), 1.0 * len(p12) / len(p2))
            overlaps.append(score)
            if score < thres_min or score > thres_max:
                continue
            
            # Calculate relative pose and essential matrix
            (t, q) = abs2relapose(im1.c, im2.c, im1.q, im2.q) # t12 is un-normalized version
            R = quat2mat(q)
            pairs.append(Namespace(im1=im1.name, im2=im2.name, 
                                   overlap=score, K1=im1.K, K2=im2.K, t=t, q=q, R=R))
    print('Total pairs {} Positive({}<overlap<{}):{}\n'.format(len(overlaps),
                                         thres_min, thres_max, len(pairs)))
    return pairs, overlaps    