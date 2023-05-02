# This Python file uses the following encoding: utf-8
import gc

from torchvision.models import resnet152
from torch import nn
import torch

from models_training.v3.TrainModel import fit_and_eval_primal, classes, load_data, eval_model


def eval_trained_resnet():
    model = resnet152(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(fr'F:\Dima\Универ\Диссертация\Проект\эксперименты\models_training\v3\models\resnet\saved_models\resnet_152_pytorch_byrn_primal_20_ep.h5'))
    _, test_ds = load_data(is_augmented=False)
    eval_model(model, test_ds)


def fit_and_eval_resnet():
    save_model_params_path_primal = fr'saved_models\resnet_152_pytorch_byrn_primal_test.h5'
    model = resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, classes)
    fit_and_eval_primal('resnet_152', model, save_model_params_path_primal)



if __name__ == '__main__':
    # fit_and_eval_resnet()
    eval_trained_resnet()
