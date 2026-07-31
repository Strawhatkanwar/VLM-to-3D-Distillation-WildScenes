import os
import yaml
import numpy as np
import cv2
import open3d as o3d
import glob
from scipy.spatial.transform import Rotation as R

def load_calibration(calib_path):
    """Loads camera intrinsics and lidar-to-camera extrinsics."""
    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)
    
    cam_data = calib['centre-camera']
    K_flat = cam_data['intrinsics']['K']
    fx, fy, cx, cy = K_flat
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    t_lidar_cam = np.array(cam_data['extrinsics']['translation'])
    q_lidar_cam = np.array(cam_data['extrinsics']['rotation']) # [qx, qy, qz, qw]
    
    r = R.from_quat(q_lidar_cam)  # [x, y, z, w]
    
    T_cam_lidar = np.eye(4)
    T_cam_lidar[:3, :3] = r.as_matrix()
    T_cam_lidar[:3, 3] = t_lidar_cam
    
    return K, T_cam_lidar

def find_matching_files(path_2d, path_3d):
  
    """Finds a matching 2D image, label, and 3D point cloud based on timestamps."""
  
  cloud_files = sorted(glob.glob(os.path.join(path_3d, '*.ply')))
    if not cloud_files:
        raise FileNotFoundError(f"No .ply files found in {path_3d}")
        
    cloud_path = cloud_files[0]
    base_ts = os.path.basename(cloud_path).replace('.ply', '')
    img_ts = base_ts.replace('.', '-')
    
    img_path = os.path.join(path_2d, 'image', f"{img_ts}.png")
    label_path = os.path.join(path_2d, 'indexLabel', f"{img_ts}.png")
    
    return cloud_path, img_path, label_path

def project_3d_to_2d(points, K, T_cam_lidar, H, W):
    
    """Projects 3D LiDAR points into the 2D camera frame and filters out-of-bounds points."""
    
    N = len(points)
    points_homo = np.hstack([points, np.ones((N, 1))])
    points_cam = (T_cam_lidar @ points_homo.T).T[:, :3]
    
    # filter points in front of camera
    front_mask = points_cam[:, 2] > 0
    points_cam_front = points_cam[front_mask]
    
    # Project to 2D pixels
    pixels_homo = (K @ points_cam_front.T).T
    pixels = pixels_homo[:, :2] / pixels_homo[:, 2:3]
    
    # Filter points inside image bounds
    in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < W) & \
                (pixels[:, 1] >= 0) & (pixels[:, 1] < H)
                
    pixels_valid = pixels[in_bounds].astype(int)
    depths_valid = points_cam_front[in_bounds][:, 2]
    valid_indices = np.where(front_mask)[0][in_bounds]
    
    return pixels_valid, depths_valid, valid_indices

def z_buffer_visibility(pixels_valid, depths_valid, H, W, tolerance=0.1):
  
    """Implements a z-buffer to keep only the front-most visible points."""
  
    depth_buffer = np.full((H, W), np.inf)
    sort_order = np.argsort(-depths_valid) # Far to near
    
    for i in sort_order:
        u, v = pixels_valid[i]
        depth_buffer[v, u] = depths_valid[i]
        
    visible = np.zeros(len(pixels_valid), dtype=bool)
    for i, (u, v) in enumerate(pixels_valid):
        if abs(depths_valid[i] - depth_buffer[v, u]) < tolerance:
            visible[i] = True
            
    return visible

def assign_2d_labels_to_3d(points, visible_pixels, visible_indices, label_map):
    """Maps 2D label values to 3D points."""
    labels_3d = np.full(len(points), -1) # -1 = unlabelled
    for i, (u, v) in enumerate(visible_pixels):
        label_val = label_map[v, u]
        labels_3d[visible_indices[i]] = label_val
    return labels_3d
