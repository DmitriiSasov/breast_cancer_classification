import shutil
from random import randrange
import os


def move_random_files_to_dir(input_dir, output_dir, count):
    moved_count = 0
    files = os.listdir(input_dir)
    while moved_count < count and len(files) > 0:
        index = randrange(len(files))
        shutil.move(os.path.join(input_dir, files[index]), os.path.join(output_dir, files[index]))
        del files[index]
        print('moved ' + str(moved_count))
        moved_count += 1


if __name__ == '__main__':
    input_dir = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\fit\Papilloma'
    output_dir_1 = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\test\Papilloma'
    output_dir_2 = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\test_2\valid\Micpap_CR'
    input_dir_2 = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\test_2\fit\Micpap_CR'
    # move_random_files_to_dir(input_dir, output_dir_1, 240)
    # move_random_files_to_dir(input_dir, output_dir_2, 416)
    move_random_files_to_dir(input_dir_2, output_dir_2, 84)
