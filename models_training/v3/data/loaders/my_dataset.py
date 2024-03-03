import os

from PIL import Image
import pandas as pd
import torch.utils.data
import numpy as np
from matplotlib import pyplot as plt


class MyDatasetLoaderWithScalars(torch.utils.data.Dataset):

    def __init__(self, scalar_csv_path, image_path, transform=None):
        self.scalars = pd.read_csv(scalar_csv_path, delimiter=";")
        self.image_path = image_path
        self.transform = transform
        self.categories = {'000': 0, '010': 1,
                           '001': 2, '011': 3,
                           '002': 4, '100': 5,
                           '101': 0 } # это некорректная разметка, так что фиг с ней пока что

    def __len__(self):
        return len(self.scalars)

    def __getitem__(self, item):
        filename = self.scalars["filename"][item]
        label = self.categories[str(self.scalars["norm"][item]) + str(self.scalars["in situ"][item])
                                + str(self.scalars["invasive"][item])]
        scalars = torch.as_tensor([float(self.scalars["epithelium"][item]), float(self.scalars["stroma"][item]),
                   float(self.scalars["adipose_tissue"][item]), float(self.scalars["background"][item]),
                   float(self.scalars["leukocyte"][item])], dtype=torch.float)
        image = Image.open(os.path.join(self.image_path, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, scalars, label

    # просмотреть некоторый набор, индексы которого будут переданы
    def view_sample(self, indexes):
        plt.figure(figsize=(len(indexes) * 4, 4))

        for i, ind in enumerate(indexes):
            image, label = self[ind]
            plt.subplot(1, len(indexes), i + 1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Type: {label}')


class MyTissueDatasetLoader(torch.utils.data.Dataset):

    def __init__(self, scalar_csv_path, image_path, transform=None):
        self.scalars = pd.read_csv(scalar_csv_path, delimiter=";")
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.scalars)

    def __getitem__(self, item):
        filename = self.scalars["filename"][item]
        label = self.scalars["epithelium"][item]
        image = Image.open(os.path.join(self.image_path, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # просмотреть некоторый набор, индексы которого будут переданы
    def view_sample(self, indexes):
        plt.figure(figsize=(len(indexes) * 4, 4))

        for i, ind in enumerate(indexes):
            image, label = self[ind]
            plt.subplot(1, len(indexes), i + 1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Type: {label}')
