import os
import shutil


def move_data_from_dir(from_dir, to_dir):
    for file in os.listdir(from_dir):
        print(file)
        shutil.copy2(os.path.join(from_dir, file), os.path.join(to_dir, file))


if __name__ == '__main__':
    from_dir = fr'F:\Dima\dissertation\Data\other_datasets\for_test\primal_data'
    to_dirs = [fr'F:\Dima\dissertation\Data\other_datasets\for_test\prepared_data\40x',
               fr'F:\Dima\dissertation\Data\other_datasets\for_test\prepared_data\100x',
               fr'F:\Dima\dissertation\Data\other_datasets\for_test\prepared_data\200x',
               fr'F:\Dima\dissertation\Data\other_datasets\for_test\prepared_data\400x']
    scale_prefixes = ['40X', '100X', '200X', '400X']
    for disease_dir in os.listdir(from_dir):
        for sob_dir in os.listdir(os.path.join(from_dir, disease_dir)):
            for i in range(4):
                image_dir = os.path.join(from_dir, disease_dir, sob_dir, scale_prefixes[i])
                to_dir_full_name = os.path.join(to_dirs[i], disease_dir)
                move_data_from_dir(image_dir, to_dir_full_name)
