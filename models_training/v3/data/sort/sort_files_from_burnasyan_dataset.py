import os
import shutil
import pandas

if __name__ == '__main__':
    dst_dir = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\fit'
    base_dir = fr'F:\Dima\dissertation\Data\other_datasets\burnasyan\burnasyan_Br'
    df = pandas.read_csv(rf'F:\Dima\dissertation\Data\other_datasets\burnasyan\burnasyan_Br.csv')
    scale_folder = 'ув4__300'
    scale_prefix = '_x4_300_'
    for name in df['Dia'].unique():
        if not os.path.exists(os.path.join(dst_dir, name)):
            os.makedirs(os.path.join(dst_dir, name))

    dirs = os.listdir(base_dir)
    for scale_dir in dirs:
        print(scale_dir)
        index = int(scale_dir) - 1
        scaled_data_dir = os.path.join(base_dir, scale_dir, scale_folder)
        dst_dir_class = df['Dia'][index]
        for file in os.listdir(scaled_data_dir):
            print(file)
            new_file_name = os.path.join(dst_dir, dst_dir_class, str(index) + scale_prefix + file)
            if os.path.isfile(new_file_name):
                print(new_file_name)
            shutil.copy2(os.path.join(scaled_data_dir, file), new_file_name)




