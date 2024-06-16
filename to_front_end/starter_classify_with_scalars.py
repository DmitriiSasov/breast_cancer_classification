# -*- coding: utf-8 -*-

import torch
from torch import device, load
from torch.cuda import is_available
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms

from to_front_end.augmentation.randstainna import RandStainNA
from to_front_end.inception_resnet_model.combined_model import CombinedModel, create_changed_inc_res_v2, ScalarInput


def get_model(model_weights_path, class_number):
    model = CombinedModel(create_changed_inc_res_v2(), ScalarInput(), class_number)
    dev = device("cuda:0" if is_available() else "cpu")
    model.to(dev)
    try:
        model.load_state_dict(load(model_weights_path, map_location=device("cuda:0" if is_available() else "cpu")))
    except FileNotFoundError:
        raise FileNotFoundError('File with model weights not found')
    model.eval()
    return model


def prepare_image(image_path: str, transformations: list):
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError('Image not found')
    except UnidentifiedImageError:
        raise UnidentifiedImageError('Incorrect image')
    transform_norm = transforms.Compose(transformations)
    img_transformed = transform_norm(img).float()
    img_transformed = img_transformed.unsqueeze_(0)
    dev = device("cuda:0" if is_available() else "cpu")
    img_transformed = img_transformed.to(dev)
    return img_transformed


def prepare_scalars(scalars: list):
    data = torch.as_tensor(list(map(float, scalars)), dtype=torch.float)
    data = data.unsqueeze_(0)
    dev = device("cuda:0" if is_available() else "cpu")
    data = data.to(dev)
    return data


def classify_image(path: str, scalars: list, model, transformations: list):
    img = prepare_image(path, transformations)
    scalars = prepare_scalars(scalars)
    output = model(img, scalars)
    return output.data.cpu().numpy().argmax()


def classify_single_image(path: str, scalars: list, weights: str, transformations: list, class_number):
    return classify_image(path, scalars, get_model(weights, class_number), transformations)


def classify_by_our_classification(image_path: str, additional_data: list):
    cancer_names = ['garbage', 'in_situ', 'invasive', 'invasive_insitu',
                    'invasive_without_surrounding_tissue', 'normal']
    try:
        model_weights = r'model_weights\test_3.h5'
        return cancer_names[
                  classify_single_image(image_path, additional_data, model_weights, [RandStainNA(
                                  yaml_file="augmentation/CRC_LAB_randomTrue_n0.yaml",
                                  std_hyper=-0.3,
                                  probability=1.0,
                                  distribution="normal",
                                  is_train=True), transforms.ToTensor()], len(cancer_names))]
    except Exception as e:
        print(e)


if __name__ == '__main__':
    i_path = r"F:\Dima\phd\test\for_ml\scalar_data_for_our_dataset\21_2064_10x_1_1.jpg"
    scalar = [1, 1, 0, 1, 0]
    print(classify_by_our_classification(i_path, scalar))
