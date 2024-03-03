import pandas
import os
import shutil

categories = {'100': 'normal', '010': 'in_situ',
              '001': 'invasive', '011': 'invasive_in_situ',
              '002': 'invasive_without_surrounding_tissue',
              '000': 'garbage'}


def sort_our_dataset(base_dir, file_with_categories):
    for key in categories:
        if not os.path.exists(os.path.join(base_dir, categories[key])):
            os.mkdir(os.path.join(base_dir, categories[key]))
    df = pandas.read_csv(file_with_categories, delimiter=";")
    for index, row in df.iterrows():
        code = str(int(row['norm'])) + str(int(row['in situ'])) + str(int(row['invasive']))
        if os.path.exists(os.path.join(base_dir, row['filename'])) and not code.__contains__("None") and code != '101':
            print(index)
            print(row['filename'])
            shutil.move(os.path.join(base_dir, row['filename']), os.path.join(base_dir, categories[code], row['filename']))


if __name__ == '__main__':
    sort_our_dataset(r'F:\Dima\phd\test\for_ml\folder_2_for_test', r'F:\Dima\phd\values\df_2_fully.csv')
