# This Python file uses the following encoding: utf-8

import timm
import torch
from torch import nn

from models_training.v3.TrainModel import fit_and_eval_primal, classes, load_data, eval_model


def eval_trained_inception_resnet():
    model = timm.create_model('inception_resnet_v2', pretrained=False)
    model.classif = nn.Linear(model.classif.in_features, classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(fr'F:\Dima\Универ\Диссертация\Проект\эксперименты\models_training\v3\models\inception_resnet\saved_models\inception_resnet_v2_pytorch_byrn_primal.h5'))
    _, test_ds = load_data(is_augmented=False)
    eval_model(model, test_ds)


def fit_and_eval_inception_resnet():
    save_model_params_path_primal = fr'saved_models\inception_resnet_v2_pytorch_byrn_primal.h5'
    model = timm.create_model('inception_resnet_v2', pretrained=True)
    model.classif = nn.Linear(model.classif.in_features, classes)
    fit_and_eval_primal('inception_resnet_v2', model, save_model_params_path_primal)


if __name__ == '__main__':
    # fit_and_eval_inception_resnet()
    eval_trained_inception_resnet()
