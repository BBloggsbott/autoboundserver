from torch import nn
from torchvision import models
import torch
from fastai.basic_data import DatasetType
import numpy as np
import os

class BuildingSegmenterNet(nn.Module):
    def __init__(self):
        super(BuildingSegmenterNet, self).__init__()
        self.fcn = models.segmentation.fcn_resnet101(pretrained=False).train()
        self.out_layer = nn.Conv2d(21, 2, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        x = self.fcn(x)
        return self.out_layer(x['out'])

def train_batch(batch, model, optimizer, loss):
    inp = batch[0].float()
    inp.requires_grad=True
    preds = model(inp)
    targets = batch[1].float()
    targets.requires_grad=True
    preds = torch.argmax(preds, dim=1).unsqueeze(dim=1).float()
    optimizer.zero_grad()
    batch_loss = loss(preds, targets.float())
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item()

def validation_loss(model, data, loss):
    running_loss = 0
    for _ in range(3):
        batch = data.one_batch(DatasetType.Valid)
        pred = model(batch[0])
        target = batch[1].float()
        pred = torch.argmax(pred, dim=1).unsqueeze(dim=1).float()
        valid_loss = loss(pred,target)
        running_loss += valid_loss.item()
    return running_loss/3

def train_model(data, model, optimizer, loss, epochs):
    print("Training starts")
    for i in range(epochs):
        print("Epoch {}".format(i))
        running_train_loss = 0
        for _ in range(12):
            batch = data.one_batch()
            train_loss = train_batch(batch, model, optimizer, loss)
            running_train_loss += train_loss
    avg_train_loss = running_train_loss/12
    valid_loss = validation_loss(model, data, loss)
    print("\tAverage Training Loss: {}\n\tAverage Validation Loss: {}".format(avg_train_loss, valid_loss))

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
