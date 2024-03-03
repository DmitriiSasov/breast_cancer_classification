
import torch
from torch import nn
import timm

from models_training.v3.models.inception_resnet.cancer.TrainModel import classes, load_data, eval_model, \
    fit_and_eval_with_logs_and_aug

data_dir_fit = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\benign_insitu_invasive\augmented\fit'
data_dir_test = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\benign_insitu_invasive\augmented\test'
data_dir = fr'F:\Dima\dissertation\Data\other_datasets\for_fit\burnasyan\benign_insitu_invasive\mixed'


def fit_and_eval_inception_resnet_invasive_benign_insitu():
    save_model_params_path_aug = fr'../saved_models/inception_resnet_v2_pytorch_byrn_ben_inv_ins_aug.h5'

    model = timm.create_model('inception_resnet_v2', pretrained=True)
    model.classif = nn.Linear(model.classif.in_features, classes)
    fit_and_eval_with_logs_and_aug(model, save_model_params_path_aug, fr'../logs/invasive_benign_insitu', True)


def eval_inception_resnet_invasive_benign_insitu():
    model = timm.create_model('inception_resnet_v2', pretrained=False)
    model.classif = nn.Linear(model.classif.in_features, classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(
        fr'saved_models\inception_resnet_v2_pytorch_byrn_ben_inv_ins_aug.h5'))
    _, test_ds = load_data(True, None, fr'F:\Dima\dissertation\Data\other_datasets\for_test\burnasyan_300x300_colored\08_08_08', None)
    eval_model(model, test_ds)

if __name__ == '__main__':
    #fit_and_eval_inception_resnet_invasive_benign_insitu()
    eval_inception_resnet_invasive_benign_insitu()
