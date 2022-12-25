from data_preparing.prepare_image import slice_images_in_directory, fix_image_name_for_1, write_images_statistic, \
    slice_image_fixed_size, fix_image_name_for_2_3, fix_image_name_for_4_5_6

if __name__ == '__main__':
    height = 300
    width = 300
    """r'F:\Dima\dissertation\Data\photos_bc\1', r'F:\Dima\dissertation\Data\photos_bc\2',
                          r'F:\Dima\dissertation\Data\photos_bc\3', r'F:\Dima\dissertation\Data\photos_bc\4',
                          r'F:\Dima\dissertation\Data\photos_bc\5',"""
    source_directories = [r'F:\Dima\dissertation\Data\photos_bc\6']
    """fr'F:\Dima\dissertation\Data\result\1', fr'F:\Dima\dissertation\Data\result\2',
                                 fr'F:\Dima\dissertation\Data\result\3', fr'F:\Dima\dissertation\Data\result\4',
                                 fr'F:\Dima\dissertation\Data\result\5',"""
    result_images_directories = [fr'F:\Dima\dissertation\Data\result\6']
    """fix_image_name_for_1, fix_image_name_for_2_3, fix_image_name_for_2_3, fix_image_name_for_4_5_6,
                       fix_image_name_for_4_5_6,"""
    fix_image_funcs = [fix_image_name_for_4_5_6]
    for i in range(len(source_directories)):
        slice_images_in_directory(source_directories[i], result_images_directories[i], width, height,
                                  slice_image_fixed_size, fix_image_funcs[i])
        color = [180, 178, 255]  # RGB
        diff = [75, 163, 255]
        file_name = 'transformed_data'
        write_images_statistic(result_images_directories[i], color, diff, file_name)
