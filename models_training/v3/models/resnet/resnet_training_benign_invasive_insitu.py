import torch
from torchvision.models import resnet152
from torch import nn
import timm

from models_training.v3.TrainModel import classes, fit_and_eval_augmented, load_data, eval_model, \
    fit_and_eval_with_logs_and_aug


def fit_and_eval_resnet_invasive_benign_insitu():
    save_model_params_path_aug = fr'saved_models\resnet_152_pytorch_byrn_ben_inv_ins_aug.h5'

    model = resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, classes)
    fit_and_eval_with_logs_and_aug(model, save_model_params_path_aug, fr'logs\invasive_benign_insitu', True)


def eval_resnet_invasive_benign_insitu():
    model = resnet152(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(
        fr'saved_models\resnet_152_pytorch_byrn_ben_inv_ins_aug.h5'))
    _, test_ds = load_data(True, None, fr'F:\Dima\dissertation\Data\other_datasets\for_test\burnasyan_300x300_colored\08_08_08', None)
    eval_model(model, test_ds)


if __name__ == '__main__':
    # fit_and_eval_resnet_invasive_benign_insitu()
    eval_resnet_invasive_benign_insitu()

