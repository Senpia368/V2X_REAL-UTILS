import numpy as np
import open3d as o3d
import yaml
from utils import boxes_to_corners_3d

def convert_bbox_to_lidar(obb, lidar_pose):
    # lidar_pose = [x, y, z, roll, pitch, yaw] in world coords
    lx, ly, lz, lroll, lpitch, lyaw = lidar_pose
    # Build transform from LIDAR -> WORLD
    T = np.eye(4)
    # R_lidar = o3d.geometry.get_rotation_matrix_from_xyz(np.radians([lroll, lpitch, lyaw]))
    R_lidar = o3d.geometry.get_rotation_matrix_from_xyz(np.radians([0, 0, lyaw]))

    T[:3,:3] = R_lidar
    T[:3, 3] = [lx, ly, lz]
    # Invert to get WORLD -> LIDAR
    T_inv = np.linalg.inv(T)

    # Convert bounding box center
    center_w = np.array([*obb.center, 1.0])
    center_l = T_inv @ center_w

    # Convert bounding box orientation
    R_obb_world = obb.R
    R_lidar_inv = R_lidar.T  # since rotation is orthonormal
    R_obb_lidar = R_lidar_inv @ R_obb_world

    # Create new OBB in lidar coords
    new_obb = o3d.geometry.OrientedBoundingBox()
    new_obb.center = center_l[:3]
    new_obb.extent = obb.extent
    new_obb.R = R_obb_lidar
    return new_obb

def visualize_bboxes(yaml_path, bin_path):
    # Load LiDAR
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Load yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    colors = {
    'Pedestrian': [1,0,0], 
    'Car': [0,1,0], 
    'TrashCan': [0,0,1], 
    'Bus': [1,0,0], 
    'ScooterRider': [1,1,0], 
    'Truck': [1,0,1], 
    'BicycleRider': [1,0.5,0], 
    'Van': [0.3,0,0], 
    'MotorcyleRider': [0,0.3,0], 
    'Scooter': [0,0,0.3], 
    'ConstructionCart': [0,0.3,0.3], 
    'LongVehicle': [0.3,0,0.3]
}

    # Parse vehicles
    boxes = []
    for vid, vdata in data['vehicles'].items():
        roll, pitch, yaw = vdata['angle']
        angle_deg = np.array([0,0,yaw], dtype=np.float32) 
        angle_rad = np.radians(angle_deg)
        location = np.array(vdata['location'], dtype=np.float32)
        extent = np.array(vdata['extent'], dtype=np.float32)  # extent is in (length, width, height) format

        # Create oriented bounding box (using center, extent, rotation)
        obb = o3d.geometry.OrientedBoundingBox()
        obb.center = location
        # Convert extent to half sizes
        obb.extent = extent * 2
        # Rotation from angles (roll, pitch, yaw)
        R = o3d.geometry.get_rotation_matrix_from_xyz(angle_rad)
        obb.R = R
        # obb_lidar = convert_bbox_to_lidar(obb, data['lidar_pose'])
        obb_lidar = obb

        # Convert OBB to lines and color them
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb_lidar)
        color = colors[vdata['obj_type']]
        lineset.colors = o3d.utility.Vector3dVector([color] * len(lineset.lines))

        boxes.append(lineset)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().show_coordinate_frame = True
    vis.add_geometry(pcd)
    for b in boxes:
        vis.add_geometry(b)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # set black background
    vis.run()
    vis.destroy_window()

    print(len(boxes), 'boxes visualized')

if __name__ == '__main__':
    bin_file = r"2023-03-17-15-53-02_1_0\-1\000012.bin"  # Replace with your .bin file path
    yaml_file = r"2023-03-17-15-53-02_1_0\-1\000012.yaml"  # Replace with your .yaml file path
    visualize_bboxes(yaml_file, bin_file)