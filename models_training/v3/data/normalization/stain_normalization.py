import PIL.Image as Image
import os
from torchvision import transforms as transforms
import cv2
import numpy as np
from skimage import color




class StainNormalization:

    def __init__(self, tmp_path: str):
        self.template_path = tmp_path

    def quick_loop(self, image, image_avg, image_std, temp_avg, temp_std, is_hed=False):

        image = (image - np.array(image_avg)) * (
                np.array(temp_std) / np.array(image_std)
        ) + np.array(temp_avg)
        if is_hed:  # HED in range[0,1]
            pass
        else:  # LAB/HSV in range[0,255]
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def getavgstd(self, image):
        avg = []
        std = []
        image_avg_l = np.mean(image[:, :, 0])
        image_std_l = np.std(image[:, :, 0])
        image_avg_a = np.mean(image[:, :, 1])
        image_std_a = np.std(image[:, :, 1])
        image_avg_b = np.mean(image[:, :, 2])
        image_std_b = np.std(image[:, :, 2])
        avg.append(image_avg_l)
        avg.append(image_avg_a)
        avg.append(image_avg_b)
        std.append(image_std_l)
        std.append(image_std_a)
        std.append(image_std_b)
        return (avg, std)

    def reinhard_cn(self, image, temp_path, color_space=None):
        isHed = False
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        template = cv2.imread(temp_path)  ### template images

        if color_space == "LAB":
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)  # LAB range[0,255]
            template = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
        elif color_space == "HED":
            isHed = True
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # color.rgb2hed needs RGB as input
            template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

            cv_image = color.rgb2hed(cv_image)  # HED range[0,1]
            template = color.rgb2hed(template)
        elif color_space == "HSV":
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        elif color_space == "GRAY":
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            return Image.fromarray(cv_image)

        image_avg, image_std = self.getavgstd(cv_image)
        template_avg, template_std = self.getavgstd(template)

        # Reinhard's Method to Stain Normalization
        cv_image = self.quick_loop(
            cv_image, image_avg, image_std, template_avg, template_std, is_hed=isHed
        )

        if color_space == "LAB":
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_LAB2RGB)
        elif color_space == "HED":  # HED[0,1]->RGB[0,255]
            cv_image = color.hed2rgb(cv_image)
            imin = cv_image.min()
            imax = cv_image.max()
            cv_image = (255 * (cv_image - imin) / (imax - imin)).astype("uint8")
        elif color_space == "HSV":
            cv_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        return Image.fromarray(cv_image)

    def __call__(self, img):
        return self.reinhard_cn(img, self.template_path)

if __name__ == "__main__":
    img_path_list = [
        "./visualization/origin/TUM-AEPINLNQ.png",
        "./visualization/origin/TUM-DFGFFNEY.png",
        "./visualization/origin/TUM-EWFNFSQL.png",
        "./visualization/origin/TUM-TCGA-CVATFAAT.png",
    ]
    template_path = "./visualization/origin/TUM-EWFNFSQL.png"
    save_dir_path = "./visualization/stain_normalization"
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    for img_path in img_path_list:
        save_path = save_dir_path + "/{}".format(img_path.split("/")[-1])
        img_colorNorm = reinhard_cn(
            img_path, template_path, save_path, isDebug=False, color_space="LAB"
        )
