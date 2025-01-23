import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R
from crop_lidar_objects import get_box_corners

def load_bin_file(file_path, num_features=3):
    """
    Load LiDAR point cloud data from a .bin file.

    Args:
    file_path (str): Path to the .bin file.
    num_features (int): Number of features per point (e.g., 3 for x, y, z or 4 for x, y, z, intensity).

    Returns:
    np.ndarray: Array of point cloud data.
    """
    data = np.fromfile(file_path, dtype=np.float32)
    return data.reshape(-1, num_features)[:, :3]  # Only use x, y, z

def load_yaml_file(yaml_path):
    """
    Load pose data from a .yaml file.

    Args:
    yaml_path (str): Path to the .yaml file.

    Returns:
    dict: Dictionary containing lidar_pose and true_ego_pose.
    """
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)
    


def visualize_point_cloud_with_poses(points, lidar_pose, true_ego_pose):
    """
    Visualize the LiDAR point cloud data with poses using Open3D.

    Args:
    points (np.ndarray): Array of point cloud data (x, y, z).
    lidar_pose (list): List representing the lidar pose [x, y, z, roll, pitch, yaw].
    true_ego_pose (list): List representing the true ego pose [x, y, z, roll, pitch, yaw].
    """
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Create pose markers (spheres for position)
    lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    lidar_sphere.translate(lidar_pose[:3])
    lidar_sphere.paint_uniform_color([1, 0, 0])  # Red for lidar_pose

    ego_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    ego_sphere.translate(true_ego_pose[:3])
    ego_sphere.paint_uniform_color([0, 0, 1])  # Blue for true_ego_pose

    # Visualize
    # o3d.visualization.draw_geometries([pcd, lidar_sphere, ego_sphere])
    # o3d.visualization.draw_geometries([pcd, lidar_sphere])
    # o3d.visualization.draw_geometries([pcd, ego_sphere])
    o3d.visualization.draw_geometries([pcd])


def visualize_pcd_with_bbox(points, bbox, color=[1, 0, 0]):
    """
    Visualize the LiDAR point cloud data with bounding box using Open3D.

    Args:
    points (np.ndarray): Array of point cloud data (x, y, z).
    bbox (list): List representing the bounding box [x, y, z, dx, dy, dz, yaw].
    color (list): List representing the RGB color [r, g, b].
    """
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Create bounding box
    bbox_center = bbox[:3]
    bbox_extent = bbox[3:6]
    bbox_yaw = bbox[6]
    bbox_corners = get_box_corners(bbox_center, bbox_extent, bbox_yaw)
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom rectangle
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top rectangle
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connections
    ]
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bbox_corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd, line_set])







if __name__ == "__main__":
    bin_file_path = r"V2X-Real\data\v2xreal\train\2023-03-17-15-53-02_1_0\-2\000000.bin"  # Replace with your .bin file path
    yaml_file_path = r"V2X-Real\data\v2xreal\train\2023-03-17-15-53-02_1_0\-2\000000.yaml"  # Replace with your .yaml file path
    
    # Adjust num_features based on your .bin file format
    num_features = 4  # For example, x, y, z, intensity

    bbox = [67,-169,2,2,1,0.7,2.3]
    
    points = load_bin_file(bin_file_path, num_features)
    poses = load_yaml_file(yaml_file_path)
    visualize_point_cloud_with_poses(points, poses['lidar_pose'], poses['true_ego_pose'])
    visualize_pcd_with_bbox(points, bbox)
    
