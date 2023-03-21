import os
from PIL import Image, ImageEnhance
import albumentations as A
import cv2
from albumentations import RandomBrightnessContrast


def horizontal_flip(image_full_name):
    return Image.open(image_full_name).transpose(Image.FLIP_LEFT_RIGHT)


def vertical_flip(image_full_name):
    return Image.open(image_full_name).transpose(Image.FLIP_TOP_BOTTOM)


# 0.2 - делает более фиолетовым, 0.0 - делает более розовым
def change_brightness_contrast(image_full_name):
    # transform = A.Compose([
    #     A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=2, p=1),
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

    # 1.4
    factor = 1
    img = enchancer.enhance(factor)

    enchancer = ImageEnhance.Color(img)

    factor = 0.7
    img = enchancer.enhance(factor)

    enchancer = ImageEnhance.Brightness(img)

    # 1.2
    factor = 1.1
    img = enchancer.enhance(factor)
    return img


def rotate_by(image_full_name, degrees):
    im = Image.open(image_full_name)
    return im.rotate(degrees)


def resize(image_full_name, height, width):
    im = Image.open(image_full_name)
    return im.resize((height, width))


def vertical_flip_images_from_dir(directory):
    for file in os.listdir(directory):
        new_file_prefix = '_ver_flip_'
        vertical_flip(os.path.join(directory, file)).save(os.path.join(directory, new_file_prefix + file))


def horizontal_flip_images_from_dir(directory):
    for file in os.listdir(directory):
        new_file_prefix = '_hor_flip_'
        horizontal_flip(os.path.join(directory, file)).save(os.path.join(directory, new_file_prefix + file))


def change_brightness_contrast_images_from_dir(directory):
    for file in os.listdir(directory):
        new_file_prefix = '_bri_con_'
        # cv2.imwrite(os.path.join(directory, new_file_prefix + file),
        #             change_brightness_contrast(os.path.join(directory, file)))
        change_brightness_contrast(os.path.join(directory, file)) \
            .save(os.path.join(directory, new_file_prefix + file))
        print(os.path.join(directory, new_file_prefix + file))


def rotate_images_from_dir(directory):
    for file in os.listdir(directory):
        new_file_prefix1 = '_rot_90'
        new_file_prefix2 = '_rot_180'
        new_file_prefix3 = '_rot_270'
        rotate_by(os.path.join(directory, file), 90).save(os.path.join(directory, new_file_prefix1 + file))
        rotate_by(os.path.join(directory, file), 180).save(os.path.join(directory, new_file_prefix2 + file))
        # rotate_by(os.path.join(directory, file), 270).save(os.path.join(directory, new_file_prefix3 + file))


def resize_images_from_dir(directory):
    for file in os.listdir(directory):
        resize(os.path.join(directory, file), 200, 200).save(os.path.join(directory, file))


def resize_images():
    folders = ['valid', 'test', 'fit']
    cancers = ['CR', 'DCIS', 'FA', 'FCD', 'Lob_CR', 'Medul_CR', 'Micpap_CR', 'Muc_CR', 'Pap_CR', 'Papilloma']
    for folder in folders:
        for cancer in cancers:
            resize_images_from_dir(
                fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\200x200\{folder}\{cancer}')


if __name__ == '__main__':
    dir = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\test_2\valid\Micpap_CR'
    # vertical_flip_images_from_dir(dir)
    # horizontal_flip_images_from_dir(dir)
    rotate_images_from_dir(dir)
    # change_brightness_contrast_images_from_dir(dir)
    # resize_images()
