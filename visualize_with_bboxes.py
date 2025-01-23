import numpy as np
import open3d as o3d
from pathlib import Path
import yaml
from scipy.spatial.transform import Rotation

def create_transformation_matrix(pose):
    """Create 4x4 transformation matrix from pose [x,y,z,rx,ry,rz]."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    return T

def transform_points(points, pose):
    """Transform points from world to sensor coordinate system."""
    T = create_transformation_matrix(pose)
    R_inv = T[:3, :3].T
    t_inv = -R_inv @ T[:3, 3]
    transformed_points = (R_inv @ points.T).T + t_inv
    return transformed_points

def get_box_corners(center, extent, angle):
    """Get corners of 3D bounding box."""
    l, w, h = extent[0]/2, extent[1]/2, extent[2]/2
    corners = np.array([
        [ l,  w,  h], [ l,  w, -h], [ l, -w,  h], [ l, -w, -h],
        [-l,  w,  h], [-l,  w, -h], [-l, -w,  h], [-l, -w, -h]
    ])
    R = Rotation.from_euler('xyz', angle).as_matrix()
    corners = (R @ corners.T).T + center
    return corners

def transform_object_info_to_lidar(obj_info, lidar_pose, true_ego_pose):
    """Transform object location and angle from ego/world frame to LiDAR frame."""
    T_ego = create_transformation_matrix(true_ego_pose)
    T_lidar = create_transformation_matrix(lidar_pose)
    T_lidar_inv = np.linalg.inv(T_lidar)
    center_h = np.array([*obj_info['location'], 1.0])
    center_lidar = (T_lidar_inv @ (T_ego @ center_h))[:3]
    obj_angle_ego = obj_info['angle']
    R_ego_to_lidar = T_lidar_inv[:3,:3] @ T_ego[:3,:3]
    delta_angle = Rotation.from_matrix(R_ego_to_lidar).as_euler('xyz')
    angle_lidar = np.array(obj_angle_ego) + delta_angle
    angle_lidar = np.deg2rad(angle_lidar)
    transformed_info = {
        'location': center_lidar,
        'extent': obj_info['extent'],
        'angle': angle_lidar,
        'obj_type': obj_info['obj_type']
    }
    return transformed_info

def visualize_objects(points, objects_info, lidar_pose, true_ego_pose):
    """Visualize objects with bounding boxes."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.transform(create_transformation_matrix(lidar_pose))
    geometries = [pcd]
    for obj_info in objects_info:
        lidar_obj_info = transform_object_info_to_lidar(obj_info, lidar_pose, true_ego_pose)
        corners = get_box_corners(
            lidar_obj_info['location'],
            lidar_obj_info['extent'],
            lidar_obj_info['angle']
        )
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom rectangle
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top rectangle
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connections
        ]
        colors = [[1, 0, 0] for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # Set background to black
    vis.run()
    vis.destroy_window()

def process_frame(bin_path, yaml_path):
    """Process a single frame, visualizing all objects."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    lidar_pose = yaml_data['lidar_pose']
    true_ego_pose = yaml_data.get('true_ego_pose', lidar_pose)
    objects_info = yaml_data['vehicles'].values()
    visualize_objects(points, objects_info, lidar_pose, true_ego_pose)

if __name__ == "__main__":
    bin_file = r"V2X-Real\data\v2xreal\train\2023-04-04-14-04-53_21_1\-1\000008.bin"  # Replace with your .bin file path
    yaml_file = r"V2X-Real\data\v2xreal\train\2023-04-04-14-04-53_21_1\-1\000008.yaml"  # Replace with your .yaml file path
    process_frame(bin_file, yaml_file)