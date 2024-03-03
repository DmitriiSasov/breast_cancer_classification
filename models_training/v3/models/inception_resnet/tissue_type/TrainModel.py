import gc
import os
import time
import copy
from itertools import cycle

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch import nn, optim
from typing import Tuple

from models_training.v3.data.augmentation.randstainna import RandStainNA
from models_training.v3.data.loaders.my_dataset import MyDatasetLoaderWithScalars

batch_size = 128
classes = 6
epochs = [20, 30, 15]
train = 0.7
test = 0.15
valid = 0.15
learning_rate = [0.001, 0.00001]
target_names = ['garbage', 'in_situ', 'invasive', 'invasive_insitu', 'invasive_without_surrounding_tissue', 'normal']  #
loss = nn.CrossEntropyLoss()


def train_model(model, criterion, optimizer, scheduler, train_ds, device, writer, num_epochs=15):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    phases = ['train', 'val']
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_for_ep, valid_for_ep = random_split(train_ds,
                                                  [int(round(len(train_ds) * 0.85)), int(round(len(train_ds) * 0.15))])
        dataloaders = {
            'train': DataLoader(train_for_ep, batch_size=batch_size, shuffle=True),
            'val': DataLoader(valid_for_ep, batch_size=batch_size, shuffle=True)
        }
        dataset_sizes = {
            'train': len(train_for_ep),
            'val': len(valid_for_ep)
        }

        # Each epoch has a training and validation phase
        metrics = {phases[0]: {}, phases[1]: {}}
        for phase in phases:
            y_pred = []
            y_true = []
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for images, scalars, labels in dataloaders[phase]:  #
                y_true.extend(labels.tolist())
                images = images.to(device)
                scalars = scalars.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, scalars)  #
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    y_pred.extend(preds.tolist())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0.0)
            metrics[phase]['loss'] = epoch_loss
            metrics[phase]['accuracy'] = epoch_acc
            metrics[phase]['prec'] = report['macro avg']['precision']
            metrics[phase]['recall'] = report['macro avg']['recall']
            metrics[phase]['f1'] = report['macro avg']['f1-score']

            print('{} Loss: {:.4f} Acc: {:.4f} SkL Acc: {:.4f} SkL Prec: {:.4f} SkL Rec: {:.4f} SkL F1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, report['accuracy'], report['macro avg']['precision'],
                report['macro avg']['recall'], report['macro avg']['f1-score']))

            print(
                confusion_matrix(y_true, y_pred)
            )

            # deep copy the model
            if phase == 'val' and report['macro avg']['f1-score'] > best_f1:
                best_f1 = report['macro avg']['f1-score']
                best_model_wts = copy.deepcopy(model.state_dict())
                writer.add_scalar('best f1', best_f1, epoch)

        writer.add_scalars('loss', {
            f'{phases[0]} loss': metrics[phases[0]]['loss'],
            f'{phases[1]} loss': metrics[phases[1]]['loss'],
        }, epoch)
        writer.add_scalars('accuracy', {
            f'{phases[0]} accuracy': metrics[phases[0]]['accuracy'],
            f'{phases[1]} accuracy': metrics[phases[1]]['accuracy'],
        }, epoch)
        writer.add_scalars('prec', {
            f'{phases[0]} prec': metrics[phases[0]]['prec'],
            f'{phases[1]} prec': metrics[phases[1]]['prec'],
        }, epoch)
        writer.add_scalars('recall', {
            f'{phases[0]} recall': metrics[phases[0]]['recall'],
            f'{phases[1]} recall': metrics[phases[1]]['recall'],
        }, epoch)
        writer.add_scalars('f1', {
            f'{phases[0]} f1': metrics[phases[0]]['f1'],
            f'{phases[1]} f1': metrics[phases[1]]['f1'],
        }, epoch)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def load_data(splitted: bool, data_dir_fit=None, data_dir_test=None, data_dir=None) -> Tuple:
    transform = transforms.Compose([RandStainNA(
        yaml_file="../CRC_LAB_randomTrue_n0.yaml",
        std_hyper=-0.3,
        probability=1.0,
        distribution="normal",
        is_train=True,
    ), transforms.ToTensor()])
    test_ds = None
    train_ds = None
    if splitted:
        if data_dir_fit is not None:
            train_ds = MyDatasetLoaderWithScalars(os.path.join(data_dir_fit, "data.csv"), data_dir_fit, transform) # datasets.ImageFolder(data_dir_fit, transform=transform)
            print(len(train_ds))
        if data_dir_test is not None:
            test_ds = MyDatasetLoaderWithScalars(os.path.join(data_dir_test, "data.csv"), data_dir_test, transform) # datasets.ImageFolder(data_dir_test, transform=transform)
            print(len(test_ds))
    else:
        full_ds = MyDatasetLoaderWithScalars(os.path.join(data_dir, "data.csv"), data_dir, transform) # datasets.ImageFolder(data_dir, transform=transform)
        print(len(full_ds))
        train_ds, test_ds = random_split(full_ds,
                                         [int(round(len(full_ds) * (train + valid))),
                                          int(round(len(full_ds) * test))]
                                         )
    return train_ds, test_ds


def eval_model(model, test_ds):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    model.eval()

    y_true = []
    y_pred = []
    for inputs, scalars, labels in test_dataloader: #
        inputs = inputs.to(device)
        scalars = scalars.to(device) #
        y_true.extend(labels.tolist())
        res = model(inputs, scalars) #
        _, preds = torch.max(res, 1)
        y_pred.extend(preds.tolist())
        del res
        del _
        del preds
        gc.collect()

    print(classification_report(y_true, y_pred, digits=4, zero_division=0.0))
    print(confusion_matrix(y_true, y_pred))

    np_y_true = np.array(y_true)
    tmp = np.zeros((np_y_true.size, np_y_true.max(initial=-1) + 1))
    tmp[np.arange(np_y_true.size), np_y_true] = 1
    y_true_one_hot = tmp
    np_y_pred = np.array(y_pred)
    tmp = np.zeros((np_y_pred.size, np_y_pred.max(initial=-1) + 1))
    tmp[np.arange(np_y_pred.size), np_y_pred] = 1
    y_pred_one_hot = tmp
    fpr = [0] * classes
    tpr = [0] * classes
    thresholds = [0] * classes
    auc_score = [0] * classes
    for i in range(classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_true_one_hot[:, i],
                                                  y_pred_one_hot[:, i])
        auc_score[i] = auc(fpr[i], tpr[i])

    print(auc_score)
    print(sum(auc_score) / classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = cycle(["red", "blue", "green", 'darkred', 'yellow', 'cyan', ])
    for class_id, color in zip(range(classes), colors):
        display = RocCurveDisplay(
            fpr=fpr[class_id],
            tpr=tpr[class_id],
            roc_auc=auc_score[class_id])
        display.plot(name=f"ROC curve for {target_names[class_id]}",
                     color=color,
                     ax=ax)

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.show()


def fit_and_eval(is_augmented: bool, writer: SummaryWriter, model, save_model_params_path):
    train_ds, test_ds = load_data(is_augmented, data_dir=fr"F:\Dima\phd\test\for_ml\scalar_data_for_our_dataset_augmented") # F:\Dima\phd\test\for_ml\my_aug

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    model = train_model(model, loss, optimizer, step_lr_scheduler, train_ds, device, writer, num_epochs=epochs[0])
    writer.close()
    torch.save(model.state_dict(), save_model_params_path)

    eval_model(model, test_ds)


def fit_and_eval_with_logs_and_aug(model, save_model_params_path, logs_dir, is_augmented):
    writer = SummaryWriter(logs_dir)
    fit_and_eval(is_augmented, writer, model, save_model_params_path)


def fit_and_eval_primal(sub_dir, model, save_model_params_path):
    writer = SummaryWriter(os.path.join(fr'logs\primal', sub_dir))
    fit_and_eval(False, writer, model, save_model_params_path)


def fit_and_eval_augmented(sub_dir, model, save_model_params_path):
    writer = SummaryWriter(os.path.join(fr'logs\augmented', sub_dir))
    fit_and_eval(True, writer, model, save_model_params_path)
