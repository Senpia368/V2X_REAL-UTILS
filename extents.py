import os
import yaml
from concurrent.futures import ProcessPoolExecutor


def process(yaml_path, dest_dir):
    yaml_path = yaml_path.replace('\\', '/')

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    frame_name = '_'.join(yaml_path.split('.')[0].split('/'))
    
    for vid, vdata in data['vehicles'].items():
        length, width, height = vdata['extent']
        label = vdata['obj_type']
        label_path = os.path.join(dest_dir, label)

        if not os.path.exists(label_path):
            os.makedirs(label_path)
            print(f'Created {label_path}')
        
        
        
        txt_path = os.path.join(label_path, f'{frame_name}_{vid}.txt')


        with open(txt_path, 'w') as f:
            f.write(f'{length} {width} {height}')

        


def write_dimensions_to_txt(yaml_dir, dest_dir):
    yaml_files = [f for f in os.listdir(yaml_dir) if f.endswith('.yaml')]
    yaml_paths = [os.path.join(yaml_dir, f) for f in yaml_files]
    print(len(yaml_paths))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process, yaml_path, dest_dir) for yaml_path in yaml_paths]
        for future in futures:
            future.result()


def check_matching_names(obj_dir, txt_dir):
    obj_files = os.listdir(obj_dir)
    txt_files = os.listdir(txt_dir)

    obj_names = [f.split('.')[0] for f in obj_files]
    txt_names = [f.split('.')[0] for f in txt_files]

    return set(obj_names) == set(txt_names)

def main():
    sequence = '2023-03-17-15-53-02_1_0'
    infra = [-1,-2]

    for infra_id in infra:
        yaml_dir = os.path.join(sequence, f'{infra_id}')
        print(f'Processing {yaml_dir}')

        write_dimensions_to_txt(yaml_dir, 'cropped_objects_txt')

    print('Names Match:', check_matching_names('cropped_objects', 'cropped_objects_txt'))

if __name__ == "__main__":
    main()




   