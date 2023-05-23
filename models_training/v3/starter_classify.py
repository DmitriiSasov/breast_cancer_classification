import timm
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms

from models_training.v3.TrainModel import classes, target_names



def get_model(model_weights_path):
    model = timm.create_model('inception_resnet_v2', pretrained=False)
    model.classif = nn.Linear(model.classif.in_features, classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    return model


def prepare_image(image_path):
    img = Image.open(image_path)
    transform_norm = transforms.Compose([transforms.ToTensor()])
    img_transformed = transform_norm(img).float()
    img_transformed = img_transformed.unsqueeze_(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_transformed = img_transformed.to(device)
    return img_transformed


if __name__ == '__main__':
    print('Input image path')
    file_path = input()
    model_weights = fr'models\inception_resnet\saved_models\inception_resnet_v2_pytorch_byrn_primal.h5'

    model = get_model(model_weights)
    img = prepare_image(file_path)
    output = model(img)
    index = output.data.cpu().numpy().argmax()
    print(target_names[index])
