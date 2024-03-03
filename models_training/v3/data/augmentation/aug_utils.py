import os
from PIL import Image, ImageEnhance
import albumentations as A
import cv2
from albumentations import RandomBrightnessContrast
import pandas as pd


def horizontal_flip(image_full_name):
    return Image.open(image_full_name).transpose(Image.FLIP_LEFT_RIGHT)


def vertical_flip(image_full_name):
    return Image.open(image_full_name).transpose(Image.FLIP_TOP_BOTTOM)


# 0.2 - делает более фиолетовым, 0.0 - делает более розовым
def change_brightness_contrast(image_full_name):
    # transform = A.Compose([
    #     A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=2, p=resnet_152),
    # ])
    #
    # # Read an image with OpenCV and convert it to the RGB colorspace
    # image = cv2.imread(image_full_name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    # # Augment an image
    # transformed = transform(image=image)
    # return transformed["image"]

    img = Image.open(image_full_name)
    enchancer = ImageEnhance.Contrast(img)

    # resnet_152.4
    factor = 0.8
    img = enchancer.enhance(factor)

    enchancer = ImageEnhance.Color(img)

    factor = 0.8
    img = enchancer.enhance(factor)

    enchancer = ImageEnhance.Brightness(img)

    # resnet_152.2
    factor = 0.8
    img = enchancer.enhance(factor)
    return img


def rotate_by(image_full_name, degrees):
    im = Image.open(image_full_name)
    return im.rotate(degrees)


def resize(image_full_name, height, width):
    im = Image.open(image_full_name)
    return im.resize((height, width))


def vertical_flip_images_from_dir(src_dir, dst_dir, table_path=None):
    table_with_data = None
    if table_path is not None:
        table_with_data = pd.read_csv(table_path, delimiter=";", dtype={'index': 'int', 'filename': 'str',
                                                                        'epithelium': 'int', 'stroma': 'int',
                                                                        'adipose_tissue': 'int',
                                                                        'background': 'int', 'leukocyte': 'int',
                                                                        'norm': 'int', 'in situ': 'int',
                                                                        'invasive': 'int'})
    for file in os.listdir(src_dir):
        new_file_prefix = 'ver_flip_'
        vertical_flip(os.path.join(src_dir, file)).save(os.path.join(dst_dir, new_file_prefix + file))
        if table_with_data is not None:
            row = table_with_data.loc[table_with_data['filename'] == file]
            row['filename'] = new_file_prefix + file
            row['index'] = len(table_with_data.index)
            table_with_data = pd.concat([table_with_data, row], ignore_index=True)
    if table_path is not None and table_with_data is not None:
        table_with_data.to_csv(table_path, sep=";", index=False)


def horizontal_flip_images_from_dir(src_dir, dst_dir, table_path=None):
    table_with_data = None
    if table_path is not None:
        table_with_data = pd.read_csv(table_path, delimiter=";", dtype={'index': 'int', 'filename': 'str',
                                                                        'epithelium': 'int', 'stroma': 'int',
                                                                        'adipose_tissue': 'int',
                                                                        'background': 'int', 'leukocyte': 'int',
                                                                        'norm': 'int', 'in situ': 'int',
                                                                        'invasive': 'int'})
    for file in os.listdir(src_dir):
        new_file_prefix = 'hor_flip_'
        horizontal_flip(os.path.join(src_dir, file)).save(os.path.join(dst_dir, new_file_prefix + file))
        if table_with_data is not None:
            row = table_with_data.loc[table_with_data['filename'] == file]
            row['filename'] = new_file_prefix + file
            row['index'] = len(table_with_data.index)
            table_with_data = pd.concat([table_with_data, row], ignore_index=True)
    if table_path is not None and table_with_data is not None:
        table_with_data.to_csv(table_path, sep=";", index=False)


def change_brightness_contrast_images_from_dir(src_dir, dst_dir):
    for file in os.listdir(src_dir):
        new_file_prefix = 'bri_con_'
        # cv2.imwrite(os.path.join(directory, new_file_prefix + file),
        #             change_brightness_contrast(os.path.join(directory, file)))
        change_brightness_contrast(os.path.join(src_dir, file)) \
            .save(os.path.join(dst_dir, new_file_prefix + file))
        print(os.path.join(dst_dir, new_file_prefix + file))


def rotate_images_from_dir(src_dir, dst_dir, table_path=None):
    table_with_data = None
    if table_path is not None:
        table_with_data = pd.read_csv(table_path, delimiter=";", dtype={'index': 'int', 'filename': 'str',
                                                                        'epithelium': 'int', 'stroma': 'int',
                                                                        'adipose_tissue': 'int',
                                                                        'background': 'int', 'leukocyte': 'int',
                                                                        'norm': 'int', 'in situ': 'int',
                                                                        'invasive': 'int'})
    for file in os.listdir(src_dir):
        prefixes = ['rot_90_', 'rot_180_']  # , 'rot_270_'
        degrees = [90, 180]  # , 270
        for new_file_prefix, degree in zip(prefixes, degrees):
            rotate_by(os.path.join(src_dir, file), degree).save(os.path.join(dst_dir, new_file_prefix + file))
            if table_with_data is not None:
                row = table_with_data.loc[table_with_data['filename'] == file]
                row['filename'] = new_file_prefix + file
                row['index'] = len(table_with_data.index)
                table_with_data = pd.concat([table_with_data, row], ignore_index=True)
    if table_path is not None and table_with_data is not None:
        table_with_data.to_csv(table_path, sep=";", index=False)


def resize_images_from_dir_with_changing_format(src_dir, dst_dir):
    for file in os.listdir(src_dir):
        res = resize(os.path.join(src_dir, file), 300, 300)
        res = res.convert("RGB")
        output_filename = file[:-3] + "jpeg"
        res.save(os.path.join(dst_dir, output_filename), "JPEG", quality=100)


def resize_images_from_dir(src_dir, dst_dir):
    for file in os.listdir(src_dir):
        resize(os.path.join(src_dir, file), 300, 300).save(os.path.join(dst_dir, file))


def resize_images():
    folders = ['Benign', 'InSitu', 'Invasive', ]
    for folder in folders:
        resize_images_from_dir(
            fr'F:\Dima\dissertation\Data\other_datasets\for_test\byrnasyan_500x500_to_300x300\{folder}',
            fr'F:\Dima\dissertation\Data\other_datasets\for_test\byrnasyan_500x500_to_300x300\{folder}')


def filter_table(table_path, files_dir):
    table = pd.read_csv(table_path, delimiter=";", dtype={'index': 'int', 'filename': 'str',
                                                          'epithelium': 'int', 'stroma': 'int',
                                                          'adipose_tissue': 'int',
                                                          'background': 'int', 'leukocyte': 'int',
                                                          'norm': 'int', 'in situ': 'int',
                                                          'invasive': 'int'})
    filenames = os.listdir(files_dir)
    filenames.remove('data.csv')
    indexes = []
    for index, row in enumerate(table.values):
        if row[1] not in filenames:
            indexes.append(index)
    table = table.drop(labels=indexes, axis=0)
    for index, row in enumerate(table.values):
        table.loc[row[0], 'index'] = index
    table.to_csv(table_path, sep=";", index=False)


if __name__ == '__main__':
    source_dir = fr'F:\Dima\phd\test\for_ml\scalar_data_for_our_dataset_augmented'
    destination_dir = fr'F:\Dima\phd\test\for_ml\scalar_data_for_our_dataset_augmented'
    table = r'F:\Dima\phd\test\for_ml\scalar_data_for_our_dataset_augmented\data.csv'
    # vertical_flip_images_from_dir(source_dir, source_dir, table)
    # horizontal_flip_images_from_dir(source_dir, source_dir, table)
    # rotate_images_from_dir(source_dir, source_dir, table)
    # change_brightness_contrast_images_from_dir(dir)
    # change_brightness_contrast_images_from_dir(src_dir, dst_dir)
    # resize_images_from_dir(src_dir, dst_dir)

    # resize_images_from_dir(r'F:\Dima\dissertation\Data\test', r'F:\Dima\dissertation\Data\test')

    filter_table(os.path.join(source_dir, 'data.csv'), source_dir)
