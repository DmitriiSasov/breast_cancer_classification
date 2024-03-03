import os
import shutil

if __name__ == '__main__':
    dst_dir = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\kaggle'
    base_dir = fr'F:\Dima\dissertation\Data\other_datasets\kaggle_breast_cancer'
    classes = ['0', 'resnet_152']

    for _class in classes:
        if not os.path.exists(os.path.join(dst_dir, _class)):
            os.makedirs(os.path.join(dst_dir, _class))

    dirs = os.listdir(base_dir)
    for class_dir in dirs:
        print(class_dir)
        if class_dir != 'IDC_regular_ps50_idx5':
            for _class in classes:
                for file in os.listdir(os.path.join(base_dir, class_dir, _class)):
                    new_file_name = os.path.join(dst_dir, _class, file)
                    if os.path.isfile(new_file_name):
                        print(new_file_name)
                    shutil.copy2(os.path.join(base_dir, class_dir, _class, file), new_file_name)
