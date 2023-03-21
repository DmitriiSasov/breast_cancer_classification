"""https://www.kaggle.com/code/songrise/implementing-resnet-18-using-keras/notebook
https://debuggercafe.com/satellite-image-classification-using-pytorch-resnet34/
https://www.kaggle.com/code/khushalbr/lyft-training-mobilenetv2/notebook
https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/
https://www.geeksforgeeks.org/how-to-set-up-and-run-cuda-operations-in-pytorch/"""

import json

import torch
import torchvision.transforms as transforms
from PIL import Image
from resnet_pytorch import ResNet 

# Open image
input_image = Image.open("Common_zebra_1.jpg")
# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# Load class names
with open("imagenet_classes.txt") as f:
    labels = [line.rstrip() for line in f]

# Classify with ResNet18
model = ResNet.from_name("resnet18", 3)
model.train()
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    logits = model(input_batch)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

print("-----")
for idx in preds:
    label = labels[idx]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    print(f"{label:<75} ({prob * 100:.2f}%)")