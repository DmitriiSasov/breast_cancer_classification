import torch
from torchvision.models import densenet121
from torch import nn

from models_training.v3.TrainModel import fit_and_eval_primal, classes, load_data, eval_model, \
    fit_and_eval_with_logs_and_aug


def fit_and_eval_densenet_invasive_benign_insitu():
    save_model_params_path_primal = fr'saved_models\densenet_121_pytorch_byrn_benign_insitu_invas_augmented.h5'
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, classes)
    fit_and_eval_with_logs_and_aug(model, save_model_params_path_primal, fr'logs\invasive_benign_insitu', True)

def eval_densenet_invasive_benign_insitu():
    model = densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(
        fr'saved_models\densenet_121_pytorch_byrn_benign_insitu_invas_augmented.h5'))
    _, test_ds = load_data(True, None, fr'F:\Dima\dissertation\Data\other_datasets\for_test\burnasyan_300x300_colored\08_08_08', None)
    eval_model(model, test_ds)


if __name__ == '__main__':
    # fit_and_eval_densenet_invasive_benign_insitu()
    eval_densenet_invasive_benign_insitu()
