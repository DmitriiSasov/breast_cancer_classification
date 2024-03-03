# This Python file uses the following encoding: utf-8

import timm
import torch
from torch.nn import Linear

from models_training.v3.models.inception_resnet.cancer.TrainModel import fit_and_eval_primal, classes, load_data, eval_model
from models_training.v3.models.inception_resnet.combined_input_inception_resnet_v2 import ScalarInput, CombinedModel, \
    ClassifierLayer
from models_training.v3.models.inception_resnet.inception_resnet_v2_arch import Inception_ResNetv2


def create_defautl_inc_res_v2_for_train():
    model = timm.create_model('inception_resnet_v2', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classif = ClassifierLayer(model.classif.in_features, classes)
    return model


def create_defautl_inc_res_v2_for_eval():
    model = timm.create_model('inception_resnet_v2', pretrained=True)
    model.classif = ClassifierLayer(model.classif.in_features, classes)
    for param in model.parameters():
        param.requires_grad = False
    return model


def create_changed_inc_res_v2():
    model = timm.create_model('inception_resnet_v2', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classif = Linear(model.classif.in_features, 10)
    return model


def create_my_i_r_v2():
    return Inception_ResNetv2(classes=classes)


def create_combined_input_model():
    return CombinedModel(create_changed_inc_res_v2(), ScalarInput(), classes)


def eval_trained_inception_resnet():
    model = create_combined_input_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(
        r'F:\Dima\Универ\Диссертация\Проект\эксперименты\models_training\v3\models\inception_resnet\saved_models\i_r_v2_our_dataset_randstainna_adam_2_layers_sigmoid_combined_input_augmented.h5'))
    _, test_ds = load_data(splitted=True, data_dir_test=r'F:\Dima\phd\test\for_ml\scalar_data_for_our_dataset_test_f2')
    eval_model(model, test_ds)


def fit_and_eval_inception_resnet():
    save_model_params_path_primal = fr'../saved_models/i_r_v2_our_dataset_randstainna_adam_2_layers_sigmoid_combined_input_augmented.h5'
    model = create_combined_input_model()
    fit_and_eval_primal('i_r_v2_our_dataset_randstainna_adam_2_layers_sigmoid_combined_input_augmented', model, save_model_params_path_primal)



if __name__ == '__main__':
    # fit_and_eval_inception_resnet()
    eval_trained_inception_resnet()
