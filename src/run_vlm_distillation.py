import os
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from projection_pipeline import (
    load_calibration, find_matching_files, 
    project_3d_to_2d, z_buffer_visibility, assign_2d_labels_to_3d
)

# Configuration
BASE_PATH = "./data/V-01/data/WildScenes"
PATH_2D = os.path.join(BASE_PATH, "WildScenes2d", "V-01")
PATH_3D = os.path.join(BASE_PATH, "Fullclouds", "V-01")
SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"

def main():
    # Load Data
    print("Loading calibration and finding matching files...")
    calib_path = os.path.join(PATH_2D, "camera_calibration.yaml")
    K, T_cam_lidar = load_calibration(calib_path)
    cloud_path, img_path, label_path = find_matching_files(PATH_2D, PATH_3D)
    
    img = cv2.imread(img_path)
    human_label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    pcd = o3d.io.read_point_cloud(cloud_path)
    points = np.asarray(pcd.points)
    H, W = img.shape[:2]
    
    # Geometric Projection & Visibility
    print("Projecting 3D points to 2D camera frame...")
    pixels_valid, depths_valid, valid_indices = project_3d_to_2d(points, K, T_cam_lidar, H, W)
    print("Running Z-Buffer visibility check...")
    visible = z_buffer_visibility(pixels_valid, depths_valid, H, W)
    visible_pixels = pixels_valid[visible]
    
    # Project Human Labels. this can be optional if you want you can or leave it. 
    labels_3d_human = assign_2d_labels_to_3d(points, visible_pixels, valid_indices[visible], human_label)
    
    # Run SAM (VLM) to generate 2D Pseudo-Labels
    print("Loading SAM model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    print("Generating SAM masks...")
    masks = mask_generator.generate(img)
    
    sam_label_map = np.zeros((H, W), dtype=np.uint8)
    for i, mask_data in enumerate(masks):
        sam_label_map[mask_data['segmentation']] = i + 1
        
    # Distill VLM Masks into 3D
    print("Distilling 2D VLM masks into 3D geometry...")
    labels_3d_sam = assign_2d_labels_to_3d(points, visible_pixels, valid_indices[visible], sam_label_map)
    
    # Visualize
    label_colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
    colors_sam = np.full((len(points), 3), 0.5) # grey default
    for i in range(len(points)):
        if labels_3d_sam[i] >= 0:
            colors_sam[i] = label_colors[labels_3d_sam[i] % len(label_colors)]
            
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    step = max(1, len(points) // 30000)
    ax.scatter(points[::step, 0], points[::step, 1], points[::step, 2], c=colors_sam[::step], s=1)
    ax.set_title('3D Cloud Painted with SAM 2D Masks (No Human Labels!)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    main()
