# -*- coding: utf-8 -*-
import sys

import timm
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms

from models_training.v3.TrainModel import classes


cancer_names = ['Неспецифицированный инвазивный рак', 'Неинвазивная протоковая карцинома', 'Фиброаденома',
                'Фиброзно-кистозные изменения', 'Дольковый инвазивный рак', 'Папиллома']


def get_model(model_weights_path):
    model = timm.create_model('inception_resnet_v2', pretrained=False)
    model.classif = nn.Linear(model.classif.in_features, classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    try:
        model.load_state_dict(torch.load(model_weights_path))
    except FileNotFoundError:
        raise FileNotFoundError('File with model weights not found')
    model.eval()
    return model


def prepare_image(image_path):
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError('Image not found')
    transform_norm = transforms.Compose([transforms.ToTensor()])
    img_transformed = transform_norm(img).float()
    img_transformed = img_transformed.unsqueeze_(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_transformed = img_transformed.to(device)
    return img_transformed


def classify_image(path, weights):
    model = get_model(weights)
    img = prepare_image(path)
    output = model(img)
    index = output.data.cpu().numpy().argmax()
    return cancer_names[index]


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Path to the image is missing')

    try:
        model_weights = fr'model_weights\inception_resnet_v2_pytorch_byrn_primal.h5'
        print(classify_image(sys.argv[1], model_weights))
    except FileNotFoundError as e:
        print(e)




