import unittest

from data_preparing.prepare_image import fix_image_name_for_1, fix_image_name_for_2_3, fix_image_name_for_4_5_6


class TestStringMethods(unittest.TestCase):

    def test_fix_name_1(self):
        expected_res = '21_2064_10x_1_1'

        res = fix_image_name_for_1('2021_2064_21_ирмж_х10_1_1')
        self.assertEqual(expected_res, res)

    def test_fix_name_2_3(self):
        expected_res = '21_1915_10x_8_1'
        res = fix_image_name_for_2_3('21_1915_ирмж_8_1')
        self.assertEqual(expected_res, res)

    def test_fix_name_4_5_6_without_year_missing(self):
        expected_res = '21_39016_10x_1_1'
        res = fix_image_name_for_4_5_6('10x_39016_21_1_1')
        self.assertEqual(expected_res, res)

    def test_fix_name_4_5_6_with_year_missing(self):
        expected_res = '21_40453_10x_64_1'
        res = fix_image_name_for_4_5_6('10x_40453_64_1')
        self.assertEqual(expected_res, res)
