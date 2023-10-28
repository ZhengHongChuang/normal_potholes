'''
Descripttion: Leetcode_code
version: 1.0
Author: zhc
Date: 2023-10-28 21:07:56
LastEditors: zhc
LastEditTime: 2023-10-28 22:36:40
'''
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
def classify_image(model, image_path, device):
    model.eval()
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.492, 0.531, 0.533],std=[0.06, 0.004, 0.053])
        ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

if __name__ == '__main__':
    model_path = "modules\\model_100.pth"
    image_path = "run\\3.jpg"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    predicted_class = classify_image(model, image_path, device)

    print(f"类别 {predicted_class}")
