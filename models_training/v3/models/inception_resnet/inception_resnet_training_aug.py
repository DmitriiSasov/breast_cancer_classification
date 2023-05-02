# This Python file uses the following encoding: utf-8

import torch
from torch import nn
import timm

from models_training.v3.TrainModel import classes, fit_and_eval_augmented, load_data, eval_model



def eval_trained_inception_resnet():
    model = timm.create_model('inception_resnet_v2', pretrained=False)
    model.classif = nn.Linear(model.classif.in_features, classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(fr'F:\Dima\Универ\Диссертация\Проект\эксперименты\models_training\v3\models\inception_resnet\saved_models\inception_resnet_v2_pytorch_byrn_augmented_20_ep.h5'))
    _, test_ds = load_data(is_augmented=True)
    eval_model(model, test_ds)


def fit_and_eval_inception_resnet():
    save_model_params_path_aug = fr'saved_models\_.h5'

    model = timm.create_model('inception_resnet_v2', pretrained=True)
    model.classif = nn.Linear(model.classif.in_features, classes)
    fit_and_eval_augmented('inception_resnet_v2', model, save_model_params_path_aug)


if __name__ == '__main__':
    fit_and_eval_inception_resnet()
