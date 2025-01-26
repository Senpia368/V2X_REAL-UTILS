import numpy as np
import open3d as o3d
import yaml
import os

from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor


# -----------
# UTILITIES
# -----------

def create_bbox_line_set(corners, color=[1, 0, 0]):
    """
    Given 8 corners of a box (N=8, shape=(8,3)),
    create an Open3D LineSet for visualization.
    """
    # Define the 12 edges of a 3D bounding box by the corner indices
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set


def get_3d_box_corners(center, lwh, roll, pitch, yaw):
    """
    Build the 8 corners (x,y,z) of a 3D bounding box in *world* coordinates,
    given:
      - center = [cx, cy, cz]
      - lwh = [length, width, height]
      - roll, pitch, yaw in degrees

    Returns: corners of shape (8,3).
    """
    l, w, h = lwh
    

    # Local box corners: [-l/2, l/2], [-w/2, w/2], [-h/2, h/2]
    #  4 top corners and 4 bottom corners
    # x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    # y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    # z_corners = [-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2]

    x_corners = [ l,  l, -l, -l,  l,  l, -l, -l]
    y_corners = [ w, -w, -w,  w,  w, -w, -w,  w]
    z_corners = [-h, -h, -h, -h,  h,  h,  h,  h]

    corners_local = np.vstack((x_corners, y_corners, z_corners)).T  # (8,3)

    # Apply roll/pitch/yaw rotation in degrees around (x,y,z) axes
    # Using scipy's R.from_euler:
    rmat = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    corners_rotated = corners_local @ rmat.T  # shape (8,3)

    # Translate to the box center
    corners_world = corners_rotated + center

    return corners_world


def transform_points(points, T):
    """
    Transform an array of 3D points (N,3) by a 4x4 matrix T.
    Returns the transformed points of shape (N,3).
    """
    n = points.shape[0]
    hom = np.hstack([points, np.ones((n,1))])  # (N,4)
    pts_trans = hom @ T.T                     # (N,4)
    return pts_trans[:, :3]


def x_to_world(pose):
    """
    The transformation matrix from a pose = [x, y, z, roll, pitch, yaw]
    into the world coordinate frame.  (CARLA-like definition.)
    """
    x, y, z, roll, yaw, pitch = pose[:]
    # Note: be sure you match the correct order for roll/pitch/yaw.
    # The function below is just an example and can be adjusted to your definition.
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation
    # The order here is roll (x), yaw (z), pitch (y) or whichever matches your system
    # Adjust carefully for your actual coordinate system.
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def get_world_to_lidar(lidar_pose):
    """
    Invert the LiDAR-pose-to-world transform
    to get a world_to_lidar transform matrix.
    """
    lidar_to_world = x_to_world(lidar_pose)   # 4x4
    world_to_lidar = np.linalg.inv(lidar_to_world)
    return world_to_lidar


# -----------
# MAIN SCRIPT
# -----------

def visualize_bboxes(yaml_path, bin_path, visualize=True):
    # Load LiDAR points
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Load yaml
    import yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # A color table for bounding boxes
    colors = {
        'Pedestrian':        [1.0, 0.0, 0.0],
        'Car':               [0.0, 1.0, 0.0],
        'TrashCan':          [0.0, 0.0, 1.0],
        'Bus':               [1.0, 0.0, 0.0],
        'ScooterRider':      [1.0, 1.0, 0.0],
        'Truck':             [1.0, 0.0, 1.0],
        'BicycleRider':      [1.0, 0.5, 0.0],
        'Van':               [0.3, 0.0, 0.0],
        'MotorcyleRider':    [0.0, 0.3, 0.0],
        'Scooter':           [0.0, 0.0, 0.3],
        'ConstructionCart':  [0.0, 0.3, 0.3],
        'LongVehicle':       [0.3, 0.0, 0.3]
    }

    # Precompute the world->LiDAR transform
    # data['lidar_pose'] must be in [x,y,z, roll, pitch, yaw] order
    world_to_lidar = get_world_to_lidar(data['lidar_pose'])

    boxes = []
    vid_corners = {}
    for vid, vdata in data['vehicles'].items():
        roll, yaw, pitch = vdata['angle']         # degrees
        length, width, height = vdata['extent']   # L,W,H
        location = np.array(vdata['location'], dtype=np.float32)  # [x, y, z] in world coords

        # 1) Get 8 corners of the bounding box in WORLD coordinates
        corners_world = get_3d_box_corners(
            center=location,
            lwh=[length, width, height],
            roll=roll,
            pitch=pitch,
            yaw=yaw
        )

        # 2) Transform corners into LIDAR coords
        corners_lidar = transform_points(corners_world, world_to_lidar)
        vid_corners[vid] = corners_lidar
        # 3) Create a line set from these corners
        obj_type = vdata['obj_type']
        color = colors[obj_type] if obj_type in colors else [1,1,1]
        lineset = create_bbox_line_set(corners_lidar, color)

        boxes.append(lineset)

    if visualize:
        # Visualize in Open3D
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True
        vis.add_geometry(pcd)
        for b in boxes:
            vis.add_geometry(b)
        
        vis.run()
        vis.destroy_window()

        print(len(boxes), 'boxes visualized')
    return vid_corners

def close_window(vis):
                print("Enter key pressed. Closing window.")
                vis.close()
                return False

def extract_cropped_objects_from_file(pcd, vid_corners, vid_type_dict, show=True, save=False, src_dir='v2xreal', save_dir='cropped_objects'):
    """
    For each bounding box corners array in 'box_corner_list':
      1) Create an OrientedBoundingBox from the corners
      2) Crop 'pcd' to that box
      3) Open an Open3D viewer showing only that cropped object

    Arguments:
      pcd              : open3d.geometry.PointCloud in LiDAR coords
      box_corner_list  : list of (8,3) arrays, each the 8 corners of one bounding box
    """
    src_dir = src_dir.replace('\\', '/')
    src_dir = '_'.join(src_dir.split('.')[0].split('/'))
    for vid, corners in vid_corners.items():
        # Convert corners to an Open3D OrientedBoundingBox
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(corners)
        )
        cropped_pcd = pcd.crop(obb)

        print(f"[{vid_type_dict[vid]} {vid}] {len(cropped_pcd.points)} points inside this bounding box.")
        
        # If empty, skip
        if len(cropped_pcd.points) == 0:
            continue
        if show:
            # Visualize in a dedicated window
            # o3d.visualization.draw_geometries([cropped_pcd])
            
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()
            vis.add_geometry(cropped_pcd)
            
        
            vis.register_key_callback(257, close_window)
            
            
            vis.run()
            vis.destroy_window()
        if save:
            # Save to a file
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dest_dir = os.path.join(save_dir, vid_type_dict[vid])
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            filename = os.path.join(dest_dir, f"{src_dir}_{vid}.pcd")
            print(f"Saving to {filename}")
            o3d.io.write_point_cloud(filename, cropped_pcd)

def get_vid_type_mapping(yaml_path):
    '''
    Get the mapping of vehicle id to vehicle type
    '''
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    vid_type_mapping = {}
    for vid, vdata in data['vehicles'].items():
        vid_type_mapping[vid] = vdata['obj_type']
    return vid_type_mapping

def crop_and_save_objects_pcd(dir_path, dest_dir):
    '''
    Crop and save objects from all .bin files in a directory
    '''
    for filename in os.listdir(dir_path):
        if filename.endswith(".bin"):
            bin_path = os.path.join(dir_path, filename)
            yaml_path = os.path.join(dir_path, filename.replace(".bin", ".yaml"))
            vid_type_dict = get_vid_type_mapping(yaml_path)
            vid_corners = visualize_bboxes(yaml_path, bin_path, False)
            pcd = points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            extract_cropped_objects_from_file(pcd, vid_corners, vid_type_dict, False, True, bin_path, os.path.join(dest_dir, "cropped_objects"))

def visualize_pcd(pcd, title=""):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)


    vis.register_key_callback(257, close_window)
    
    vis.run()
    vis.destroy_window()

def process_file(bin_path, yaml_path, dest_dir):
    vid_type_dict = get_vid_type_mapping(yaml_path)
    vid_corners = visualize_bboxes(yaml_path, bin_path, False)
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    extract_cropped_objects_from_file(pcd, vid_corners, vid_type_dict, False, True, bin_path, dest_dir)

def crop_and_save_objects_pcd_parallel(dir_path, dest_dir):
    '''
    Crop and save objects from all .bin files in a directory
    '''
    with ProcessPoolExecutor() as executor:
        for filename in os.listdir(dir_path):
            if filename.endswith(".bin"):
                bin_path = os.path.join(dir_path, filename)
                yaml_path = os.path.join(dir_path, filename.replace(".bin", ".yaml"))
                executor.submit(process_file, bin_path, yaml_path, dest_dir)
    
if __name__ == '__main__':
    # bin_path = r"2023-03-17-15-53-02_1_0\-2\000006.bin"  # Replace with your .bin file path
    # yaml_path = r"2023-03-17-15-53-02_1_0\-2\000006.yaml"  # Replace with your .yaml file path
    # corners = visualize_bboxes(yaml_path, bin_path, True)
    # vid_type_dict = get_vid_type_mapping(yaml_path)
    # pcd = points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # extract_cropped_objects_from_file(pcd, corners, vid_type_dict)
    # crop_and_save_objects_pcd("2023-03-17-15-53-02_1_0/-2", dest_dir='')
    crop_and_save_objects_pcd_parallel("2023-03-17-15-53-02_1_0/-1", dest_dir='cropped_objects')
    for folder in os.listdir('cropped_objects'):
        print(f'{folder}: {len(os.listdir(os.path.join("cropped_objects", folder)))}')
    #     if os.path.isdir(os.path.join('cropped_objects', folder)):
    #         for filename in os.listdir(os.path.join('cropped_objects', folder)):
    #             if filename.endswith(".pcd"):
    #                 pcd = o3d.io.read_point_cloud(os.path.join('cropped_objects', folder, filename))
    #                 title = f"{folder} {filename}"
    #                 visualize_pcd(pcd, title)
                    
    #         break