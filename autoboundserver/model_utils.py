from torch import nn
import torch
from fastai.basic_data import DatasetType
import numpy as np
import os

class BuildingSegmenterNet(nn.Module):
    def __init__(self):
        super(BuildingSegmenterNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 16, (5,5)),
            nn.MaxPool2d((2,2))
        )
        self.seq2 = nn.Sequential(
            nn.Linear((126*126*16), 512),
            nn.ReLU()
        )
        self.dropout1 = nn.Dropout(0.33)
        self.seq3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.seq4 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.seq5 = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU()
        )
        self.seq6 = nn.Sequential(
            nn.Linear(512, 256*256),
            nn.ReLU()
        )
        self.seq7 = nn.Sequential(
            nn.Conv2d(1, 1, (3,3), padding = 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.seq1(x)
        x = x.view(-1, 126*126*16)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.dropout1(x)
        x = self.seq4(x)
        x = self.seq5(x)
        x = self.seq6(x)
        x = x.view(-1, 1, 256, 256)
        x = self.seq7(x)
        return x

def get_model_from_file(pretrained=True, filename=os.path.join('models', 'autoboundModel.pth')):
    if pretrained:
        model = torch.load(filename)
    else:
        model = BuildingSegmenterNet()
    return model

def predict_segmented_image(model, image_tensor, asPIL = True):
    res = model(image_tensor)
    res = torch.argmax(res, dim=1).unsqueeze(dim=1).float()
    img = res[0].numpy()[0]
    if asPIL:
        from PIL import Image
        img = Image.fromarray((img*255).astype(np.uint8))
        img = img.convert('RGB')
    return img
