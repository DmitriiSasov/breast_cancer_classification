# This is a sample Python script.
import os

import cv2
import numpy as np
from pandas import DataFrame


def read_image_with_unicode_name(file_name: str):
    stream = open(file_name, 'rb')
    content = bytearray(stream.read())
    numpy_array = np.asarray(content, dtype=np.uint8)
    bgr_image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    stream.close()
    return hsv_image


def calc_color_percentage(img_name, color: list, diff: list):
    img = read_image_with_unicode_name(img_name)

    boundaries = [([max(color[0] - diff[0], 0), max(color[1] - diff[1], 0),
                    max(color[2] - diff[2], 0)],
                   [color[0], color[1], color[2]])]

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(img, lower, upper)

        ratio_color = cv2.countNonZero(mask) / (img.size / 3)
        print('color pixel percentage:', np.round(ratio_color * 100, 2))
        return ratio_color * 100


def slice_images_in_directory(directory_name: str, result_folder_path: str, horizontal_param: int, vertical_param: int,
                              slicing_func, fix_name_func):
    files = os.listdir(directory_name)
    files = filter(lambda x: x.__contains__('.jpg') or x.__contains__('.png'), files)
    print('start slicing')
    for file_name in files:
        print(file_name)
        image_name_and_path = os.path.join(directory_name, file_name)
        img = read_image_with_unicode_name(image_name_and_path)
        img_name_and_extension = image_name_and_path.split('\\')[len(image_name_and_path.split('\\')) - 1]
        img_extension = img_name_and_extension.split('.')[1]
        img_name = img_name_and_extension.split('.')[0]
        new_images = slicing_func(img, horizontal_param, vertical_param)
        for i in range(len(new_images)):
            save_image(result_folder_path, img_name + '_' + str(i + 1), img_extension, new_images[i], fix_name_func)


def write_images_statistic(folder_path: str, color: list, color_diff: list, result_file_name: str):
    files = os.listdir(folder_path)
    files = list(filter(lambda x: x.__contains__('.jpg') or x.__contains__('.png'), files))
    tissue_percent = []
    for file_name in files:
        percent = calc_color_percentage(os.path.join(folder_path, file_name), color, color_diff)
        tissue_percent.append(percent)
    excel_data = DataFrame({'Имя файла': files, f'Процент розового цвета на картинке':
                            tissue_percent})
    excel_data.to_excel(os.path.join(folder_path, f'{result_file_name}.xlsx'), sheet_name='sheet1', index=False)


def fix_image_name_for_1(img_name: str):
    name_parts = img_name.split('_')
    return name_parts[2] + '_' + name_parts[1] + '_' + '10x' + '_' + name_parts[-2] + '_' + name_parts[-1]


def fix_image_name_for_2_3(img_name: str):
    return img_name.replace('ирмж', '10x')


def fix_image_name_for_4_5_6(img_name: str):
    name_parts = img_name.split('_')
    res = ''
    if len(name_parts) == 4:
        res = '21_' + name_parts[1] + '_' + '10x_' + name_parts[2] + '_' + name_parts[3]
    else:
        res = name_parts[2] + '_' + name_parts[1] + '_' + '10x' + '_' + name_parts[-2] + '_' + name_parts[-1]
    return res


def save_image(result_folder_path, img_name, img_extension, image, fix_name_func):
    # Меняем русские символы на английские, чтобы сохранить картинку с новым названием без русских символов
    img_name = fix_name_func(img_name)

    cv2.imwrite(result_folder_path + '\\' + img_name + "." + img_extension, cv2.cvtColor(image, cv2.COLOR_HSV2BGR))


def slice_image_into_equal_parts(img, horizontal_split: int, vertical_split: int):
    """Пока не используется"""
    img2 = img
    height, width, channels = img.shape
    new_images = []
    for ih in range(vertical_split):
        for iw in range(horizontal_split):
            x = int(width / horizontal_split * iw)
            y = int(height / vertical_split * ih)
            h = int(height / vertical_split)
            w = int(width / horizontal_split)
            img2 = img2[y:y + h, x:x + w]
            new_images.append(img2)
            img2 = img
    return new_images


def slice_image_fixed_size(img, horizontal_size: int, vertical_size: int):
    new_images = []
    height, width = img.shape[:2]
    width_remains = width - (horizontal_size * (width // horizontal_size))
    height_remains = height - (vertical_size * (height // vertical_size))
    new_images.extend(
        slice_image_into_equal_parts(
            img[0:vertical_size * (height // vertical_size), 0:horizontal_size * (width // horizontal_size)],
            width // horizontal_size, height // vertical_size))

    # определяем, нужены ли доп. фотографии с нахлестом для нижнего края картинки
    if height_remains > vertical_size / 2:
        new_images.extend(
            slice_image_into_equal_parts(
                img[height - vertical_size: height, 0:horizontal_size * (width // horizontal_size)],
                width // horizontal_size, 1))
    # определяем, нужены ли доп. фотографии с нахлестом для правого края картинки
    if width_remains > horizontal_size / 2:
        new_images.extend(slice_image_into_equal_parts(
            img[0: vertical_size * (height // vertical_size), width - horizontal_size:width],
            1, height // vertical_size))
    return new_images


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    slice_images_in_directory(r'F:\Dima\dissertation\Data\photos_bc\1', r'F:\Dima\dissertation\Data\result\1', 2, 2)
