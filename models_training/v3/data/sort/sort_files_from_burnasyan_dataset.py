import os
import shutil
import pandas
from pandas import DataFrame


def create_dirs(names, directory):
    for name in names:
        if not os.path.exists(os.path.join(directory, name)):
            os.makedirs(os.path.join(directory, name))


def copy_images_by_class_and_scale(base_dir: str, dst_dir: str, scale_folder: str, scale_prefix: str,
                                   classes_doc: DataFrame, class_column, classes):
    dirs = os.listdir(base_dir)
    for scale_dir in dirs:
        print(scale_dir)
        index = int(scale_dir) - 1
        scaled_data_dir = os.path.join(base_dir, scale_dir, scale_folder)
        dst_dir_class = classes_doc[class_column][index]
        if dst_dir_class in classes:
            for file in os.listdir(scaled_data_dir):
                print(file)
                new_file_name = os.path.join(dst_dir, dst_dir_class, str(index) + scale_prefix + file)
                if os.path.isfile(new_file_name):
                    print(new_file_name)
                shutil.copy2(os.path.join(scaled_data_dir, file), new_file_name)


def get_inamges_by_diag():
    dst_dir = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\test_3\fit'
    base_dir = fr'F:\Dima\dissertation\Data\other_datasets\burnasyan\burnasyan_Br'
    df = pandas.read_csv(rf'F:\Dima\dissertation\Data\other_datasets\burnasyan\burnasyan_Br.csv')
    scale_folder = 'ув4__300'
    scale_prefix = '_x4_300_'
    classes = list(filter(lambda x: x not in ['Cribr_CR', 'Medul_CR', 'Muc_CR', 'Pap_CR'], df['Dia2'].unique()))
    create_dirs(classes, dst_dir)

    copy_images_by_class_and_scale(base_dir, dst_dir, scale_folder, scale_prefix, df, 'Dia', classes)


def get_images_by_diag2():
    dst_dir = fr'F:\Dima\phd\test\for_ml\byrn_300_x10'
    base_dir = fr'F:\Dima\dissertation\Data\other_datasets\burnasyan\burnasyan_Br'
    df = pandas.read_csv(rf'F:\Dima\dissertation\Data\other_datasets\burnasyan\burnasyan_Br.csv')
    scale_folder = 'ув10__300'
    scale_prefix = '_x10_300_'
    classes = df['Dia2'].unique()
    create_dirs(classes, dst_dir)

    copy_images_by_class_and_scale(base_dir, dst_dir, scale_folder, scale_prefix, df, 'Dia2', classes)


if __name__ == '__main__':
    get_images_by_diag2()
