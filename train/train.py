'''
Descripttion: Leetcode_code
version: 1.0
Author: zhc
Date: 2023-10-28 20:18:56
LastEditors: zhc
LastEditTime: 2023-10-28 20:34:00
'''

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import json
from torchvision import transforms 
from PIL import Image
import torchvision.models as models


class MyDataset(Dataset):
    def __init__(self,root,transform=None):
        self.imgs = []
        for path in os.listdir(root):
            if path.startswith('normal'):
                label = 0
            elif path.startswith('potholes'):
                label = 1
            self.imgs.append((os.path.join(root,path),label))     
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.492, 0.531, 0.533],std=[0.06, 0.004, 0.053])
            ])
        else:
            self.transform = transform

    def __getitem__(self,index):
        img_path= self.imgs[index][0]
        imgLabel = self.imgs[index][1]
        data = Image.open(img_path)
        if data.mode != 'RGB':
            data = data.convert('RGB')
        data = self.transform(data)
        return data,imgLabel
    def __len__(self):
        return len(self.imgs)
    
if __name__ == '__main__':
    dataset_dir = "datasets\\data"
    batch_size = 32
    epochs = 500
    train_data = MyDataset(dataset_dir)
    val_data = MyDataset(dataset_dir)
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    resnet = models.resnet50(pretrained=True) 
    for parm in resnet.parameters():
        parm.requires_grad = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 2),
    )
    criterion = nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.0001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    resnet.to(device)
    for epoch in range(epochs):
        print('\nEpoch: %d' % (epoch + 1))

        resnet.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        if (epoch+1) % 100 == 0:
            torch.save(resnet.state_dict(), 'model/model_%d.pth' % (epoch + 1))
    
