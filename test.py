import yaml
from collections import Counter
import os

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
    
def get_objects(yaml_file_path):
    """
    Create a Counter object containing the number of each object type in the .yaml file.

    Args:
    yaml_file_path (str): Path to the .yaml file.

    Returns:
    Counter: Counter object containing the number of each object type.
    """
    yaml_data = load_yaml_file(yaml_file_path)

    objects = []

    for vehicle, attributes in yaml_data['vehicles'].items():
        objects.append(attributes['obj_type'])

    return Counter(objects)

def combine_counters(counter1, counter2):
    """
    Combine two counters.

    Args:
    counter1 (Counter): First counter.
    counter2 (Counter): Second counter.

    Returns:
    Counter: Combined counter.
    """
    return counter1 + counter2

def get_all_objects(dataset_path , cam_ids=['-1', '-2'], counter=None):
    """
    Get all objects from the dataset.

    Args:
    dataset_path (str): Path to the dataset directory.
    cam_ids (list): List of camera IDs to consider.
    counter (Counter): Counter object to update.

    Returns:
    Counter: Counter object containing the number of each object type.
    """
    if counter is None:
        counter = Counter()

    num_frames = 0

    for train in os.listdir(dataset_path):
        train_path = os.path.join(dataset_path, train)
        print(train_path)
        for seq in os.listdir(train_path):
            seq_path = os.path.join(train_path, seq)
            print(seq_path)
            for cam_id in cam_ids:
                cam_path = os.path.join(seq_path, cam_id)
                # Check if cam_path exists
                if not os.path.exists(cam_path):
                    continue
                yaml_files = [f for f in os.listdir(cam_path) if f.endswith('.yaml')]
                for yaml_file in yaml_files:
                    yaml_file_path = os.path.join(cam_path, yaml_file)
                    counter = combine_counters(counter, get_objects(yaml_file_path))
                    num_frames += 1

    return counter, num_frames
    
if __name__ == "__main__":
    # yaml_file_path = r"C:\Users\zdl551\Downloads\test (1)\test\2023-04-03-18-28-32_22_0\-1\000005.yaml"  # Replace with your .yaml file path
    # yaml_file_path2 = r"C:\Users\zdl551\Downloads\test (1)\test\2023-04-03-18-28-32_22_0\-1\000006.yaml"  # Replace with your .yaml file path

    dataset_path = 'dataset'

    object_counter, num_frames = get_all_objects(dataset_path)

    print(object_counter)
    print(num_frames)
    
    