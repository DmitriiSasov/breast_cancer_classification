import torch
from torch import nn
from torch.nn import Linear

from models_training.v3.models.inception_resnet.inception_resnet_v2_arch import Inception_ResNetv2


class ClassifierLayer(nn.Module):
    def __init__(self, in_features, classes):
        super(ClassifierLayer, self).__init__()
        self.first_linear = Linear(in_features, in_features)
        self.second_linear = Linear(in_features, classes)

    def forward(self, x):
        x = self.first_linear(x)
        x = nn.Sigmoid()(x)
        x = self.second_linear(x)
        return x


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class ScalarInput(nn.Module):
    def __init__(self):
        self.input_size = 5
        self.output_size = 10
        super(ScalarInput, self).__init__()
        self.layer = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        return self.layer(x)


class CombinedModel(nn.Module):

    def __init__(self, inception_resnet_v2: Inception_ResNetv2, scalar_input: ScalarInput, classes):
        super(CombinedModel, self).__init__()
        self.global_average_pooling_output_size = 10
        self.inception_resnet_v2 = inception_resnet_v2
        self.scalar_input = scalar_input
        self.classes = classes
        self.classifier = ClassifierLayer(self.global_average_pooling_output_size + scalar_input.output_size,
                                          self.classes)

    def forward(self, x1, x2):
        x1 = self.inception_resnet_v2(x1)
        x2 = self.scalar_input(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x
