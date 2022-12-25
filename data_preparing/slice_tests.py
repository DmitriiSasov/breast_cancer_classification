import os
import unittest

import pandas
import transliterate

from data_preparing.prepare_image import slice_images_in_directory, calc_color_percentage, slice_image_into_equal_parts, \
    write_images_statistic, slice_image_fixed_size, read_image_with_unicode_name, fix_image_name_for_1


class TestStringMethods(unittest.TestCase):
    PHOTO_SIZE_HORIZONTAL = 1920
    PHOTO_SIZE_VERTICAL = 1440

    def test_calc_color(self):
        color = [180, 178, 255]  # RGB
        diff = [80, 155, 255]
        self.assertTrue(calc_color_percentage(r'F:\Dima\dissertation\Data\photos_bc\1\2021_2064_21_ирмж_х10_6.jpg', color, diff) > 15)

    def test_calc_white_color_(self):
        color = [255, 255, 255]  # RGB
        diff = [75, 75, 75]
        self.assertTrue(
            calc_color_percentage(r'F:\Dima\dissertation\Data\result\3\21_1715_irmzh_58_2.jpg', color,
                                  diff) > 54)

    def test_slicing(self):
        file = r'../photos/белый_цвет.jpg'
        img = read_image_with_unicode_name(file)
        result_files = slice_image_into_equal_parts(img, 2, 2)
        self.assertTrue(len(result_files) == 4)

    def test_join_path(self):
        print(os.path.join('../photos', 'white_test.jpg'))

    def test_slicing_in_dir(self):
        slice_images_in_directory(r'../photos', r'../tests_results', 2, 2, slice_image_into_equal_parts, lambda x: x)

    def test_slicing_in_dir_fixed_size(self):
        slice_images_in_directory(r'../photos', r'../tests_results', 300, 300, slice_image_fixed_size, lambda x: x)

    def test_detecting_language(self):
        print(transliterate.detect_language('2021_2064_21_ирмж_х10_3'))

    def test_slicing_in_dir_many_photos(self):
        source_directory = r'F:\Dima\dissertation\Data\result\test_example\example_source'
        result_images_directory = r'F:\Dima\dissertation\Data\result\test_example\1x4'
        slice_images_in_directory(source_directory, result_images_directory, 2, 2, slice_image_into_equal_parts)
        result_files = os.listdir(result_images_directory)
        self.assertTrue(len(result_files) == ((len(os.listdir(source_directory))) * 2 * 2))

    def test_write_images_statistic(self):
        folder_path = r'F:\Dima\dissertation\Data\result\6'
        color = [180, 178, 255]  # RGB
        diff = [75, 163, 255]
        file_name = 'transformed_data'
        write_images_statistic(folder_path, color, diff, file_name)
        result_files = os.listdir(folder_path)
        xlsx_data = pandas.read_excel(os.path.join(folder_path, file_name + '.xlsx'), engine="openpyxl")
        string_numbers = len(xlsx_data.index)
        self.assertTrue(string_numbers == len(result_files) - 1)

    def test_slice_image_fixed_size(self):
        height = 300
        width = 300
        source_directory = r'F:\Dima\dissertation\Data\result\test_example\example_source'
        result_images_directory = fr'F:\Dima\dissertation\Data\result\test_example\{height}x{width}'
        slice_images_in_directory(source_directory, result_images_directory, width, height, slice_image_fixed_size, fix_image_name_for_1)
        result_files = os.listdir(result_images_directory)
        self.assertTrue(len(result_files) == ((len(os.listdir(source_directory))) *
                                              (int(round(self.PHOTO_SIZE_HORIZONTAL / width)) *
                                               (int(round(self.PHOTO_SIZE_VERTICAL / height))))))

    def test_complex_slice_image_fixed_size_with_statistic(self):
        height = 300
        width = 300
        source_directory = r'F:\Dima\dissertation\Data\result\test_example\example_source'
        result_images_directory = fr'F:\Dima\dissertation\Data\result\test_example\{height}x{width}'
        slice_images_in_directory(source_directory, result_images_directory, width, height, slice_image_fixed_size,
                                  fix_image_name_for_1)
        color = [180, 178, 255]  # RGB
        diff = [75, 163, 255]
        file_name = 'transformed_data'
        write_images_statistic(result_images_directory, color, diff, file_name)
        result_files = os.listdir(result_images_directory)
        xlsx_data = pandas.read_excel(os.path.join(result_images_directory, file_name + '.xlsx'), engine="openpyxl")
        string_numbers = len(xlsx_data.index)
        self.assertTrue(string_numbers == len(result_files) - 1)
        self.assertTrue(len(result_files) == ((len(os.listdir(source_directory))) *
                                              (int(round(self.PHOTO_SIZE_HORIZONTAL / width)) *
                                               (int(round(self.PHOTO_SIZE_VERTICAL / height)))) + 1))

if __name__ == '__main__':
    unittest.main()
