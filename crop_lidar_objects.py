import numpy as np
import open3d as o3d
from pathlib import Path
import yaml
import numpy as np
from scipy.spatial.transform import Rotation

def transform_points(points, pose):
    """Transform points from world to sensor coordinate system."""
    # Extract translation and rotation from pose
    translation = np.array(pose[:3])
    rotation = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    
    # Apply inverse transform
    R_inv = rotation.T
    t_inv = -R_inv @ translation
    
    # Transform points
    transformed_points = (R_inv @ points.T).T + t_inv
    return transformed_points

def get_box_corners(center, extent, angle):
    """Get corners of 3D bounding box."""
    # Convert extent to half-lengths
    l, w, h = extent[0]/2, extent[1]/2, extent[2]/2
    
    # Define corners in object coordinate system
    corners = np.array([
        [ l,  w,  h], [ l,  w, -h], [ l, -w,  h], [ l, -w, -h],
        [-l,  w,  h], [-l,  w, -h], [-l, -w,  h], [-l, -w, -h]
    ])
    
    # Create rotation matrix from angles
    R = Rotation.from_euler('xyz', angle).as_matrix()
    
    # Rotate corners and add center offset
    corners = (R @ corners.T).T + center
    return corners


def transform_object_info_to_lidar(obj_info, lidar_pose, true_ego_pose):
    """
    Transform object location and angle from ego/world frame to LiDAR frame.
    """
    # Create transformation matrices
    T_ego = np.eye(4)
    T_ego[:3,3] = true_ego_pose[:3]
    T_ego[:3,:3] = Rotation.from_euler('xyz', true_ego_pose[3:]).as_matrix()

    T_lidar = np.eye(4)
    T_lidar[:3,3] = lidar_pose[:3]
    T_lidar[:3,:3] = Rotation.from_euler('xyz', lidar_pose[3:]).as_matrix()
    
    # Invert LiDAR transform (from world to LiDAR)
    T_lidar_inv = np.linalg.inv(T_lidar)

    # Object center in homogeneous coords
    center_h = np.array([*obj_info['location'], 1.0])

    # Transform center to LiDAR frame
    center_lidar = (T_lidar_inv @ (T_ego @ center_h))[:3]

    # Combine angles for object orientation
    # (simplified approach: object angle in ego + difference from ego->lidar)
    # You might need to refine this based on how angles are defined
    obj_angle_ego = obj_info['angle']
    # Extract orientation difference from T_ego to T_lidar
    R_ego_to_lidar = T_lidar_inv[:3,:3] @ T_ego[:3,:3]
    # Convert to euler angles
    delta_angle = Rotation.from_matrix(R_ego_to_lidar).as_euler('xyz')
    angle_lidar = np.array(obj_angle_ego) + delta_angle

    # Update object info
    transformed_info = {
        'location': center_lidar,
        'extent': obj_info['extent'],
        'angle': angle_lidar,
        'obj_type': obj_info['obj_type']
    }
    return transformed_info

def crop_object_points(points, obj_info, lidar_pose, true_ego_pose):
    """Crop points within object bounding box."""
    # Transform points to lidar frame
    points_local = transform_points(points, lidar_pose)

    # Transform object info to lidar frame
    lidar_obj_info = transform_object_info_to_lidar(obj_info, lidar_pose, true_ego_pose)

    # Get object box corners in LiDAR frame
    corners = get_box_corners(
        lidar_obj_info['location'],
        lidar_obj_info['extent'],
        lidar_obj_info['angle']
    )

    # Create open3d box
    box = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(corners)
    )

    # Convert points to open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_local)

    # Crop points
    cropped_pcd = pcd.crop(box)
    return np.asarray(cropped_pcd.points)

def visualize_object(points, obj_info, lidar_pose, true_ego_pose, save_path=None):
    """Visualize and optionally save cropped object."""
    obj_points = crop_object_points(points, obj_info, lidar_pose, true_ego_pose)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_points)
    pcd.paint_uniform_color([1, 0, 0])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([pcd, coord_frame])

    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)

def process_frame(bin_path, yaml_path, output_dir=None):
    """Process a single frame, visualizing all objects."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]

    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    lidar_pose = yaml_data['lidar_pose']
    true_ego_pose = yaml_data.get('true_ego_pose', lidar_pose)  # fallback if missing

    for obj_id, obj_info in yaml_data['vehicles'].items():
        print(f"Processing {obj_info['obj_type']} (ID: {obj_id})")

        save_path = None
        if output_dir:
            save_path = Path(output_dir) / f"{obj_info['obj_type']}_{obj_id}.pcd"

        visualize_object(points, obj_info, lidar_pose, true_ego_pose, save_path)

if __name__ == "__main__":
    bin_file = "dataset/train1/2023-03-17-15-53-02_1_0/-2/000026.bin"
    yaml_file = "dataset/train1/2023-03-17-15-53-02_1_0/-2/000026.yaml"
    output_dir = "cropped_objects"
    
    # Create output directory if saving
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
        
    process_frame(bin_file, yaml_file, output_dir)