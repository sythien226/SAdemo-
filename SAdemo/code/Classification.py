import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models, transforms
import numpy as np
from PIL import Image

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Average', 'Fair', 'Good']

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

dir_model = "./static/models/"

def Classification(img, name_models):
    path_model = dir_model + name_models + ".pt"
    
    model_ft = models.mobilenet_v2(pretrained=True)

    num_ftrs = model_ft.classifier[1].in_features

    model_ft.classifier = nn.Sequential(
        nn.Dropout(0.7),
        nn.Linear(num_ftrs, 50),
        nn.Dropout(0.5),
        nn.Linear(50, 3)
    )

    model_ft.load_state_dict(torch.load(path_model, device))
    model_ft.eval()
    
    img1 = data_transforms['test'](img)
    img1 = img1.unsqueeze(0)
    img1 = img1.to(device)

    with torch.no_grad():
        model_ft.to(device)
        outputs = model_ft(img1)
        _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]