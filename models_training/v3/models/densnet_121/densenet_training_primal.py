# This Python file uses the following encoding: utf-8
import torch
from torchvision.models import densenet121
from torch import nn

from models_training.v3.TrainModel import fit_and_eval_primal, classes, load_data, eval_model


def eval_trained_densenet():
    model = densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(fr'F:\Dima\Универ\Диссертация\Проект\эксперименты\models_training\v3\models\densnet_121\saved_models\densenet_121_pytorch_byrn_primal.h5'))
    _, test_ds = load_data(is_augmented=False)
    eval_model(model, test_ds)


def fit_and_eval_densenet():
    save_model_params_path_primal = fr'saved_models\densenet_121_pytorch_byrn_primal.h5'
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, classes)
    fit_and_eval_primal('densenet_121', model, save_model_params_path_primal)


if __name__ == '__main__':
    # fit_and_eval_densenet()
    eval_trained_densenet()
