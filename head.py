import pandas as pd
import numpy as np
import datetime
import random

import matplotlib.pyplot as plt
import seaborn as sns

from os import listdir, mkdir, rename
from os.path import join, exists
import shutil

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

from models.resnet import ResNet
from models.cnn import SimpleCNN
from models.unet import UNet
from models.dataset import dataset

num_0 = len(listdir('../data/S2_dataset/0/RGB'))
num_1 = len(listdir('../data/S2_dataset/1/RGB'))

files_1 = listdir('../data/S2_dataset/1/RGB')
files_1_swir = listdir('../data/S2_dataset/1/SWIR')
path_rgb = '../data/S2_dataset/1/RGB/'
path_swir = '../data/S2_dataset/1/SWIR/'

test_ratio, train_ratio = 0.3, 0.7
num_test = int((num_0+num_1)*test_ratio)
num_train = int((num_0+num_1)*train_ratio)

img_size = (502, 502)
num_classes = 2

print("Number of train samples:", num_train)
print("Number of test samples:", num_test)
random.seed(231)
train_idxs = np.array(random.sample(range(num_train+num_test), num_train))
mask = np.ones(num_0+num_1, dtype=bool)
mask[train_idxs] = False

images = {}
images['RGB'] = listdir('../data/S2_dataset/10/RGB')
images['SWIR'] = listdir('../data/S2_dataset/10/SWIR')
wildfire = np.zeros(len(listdir('../data/S2_dataset/0/RGB'))).tolist()\
  +np.ones(len(listdir('../data/S2_dataset/1/RGB'))).tolist()

random.Random(231).shuffle(images['RGB'])
random.Random(231).shuffle(images['SWIR'])
random.Random(231).shuffle(wildfire)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
num_epochs = 20
criterion = nn.BCEWithLogitsLoss()


MODELS = {'UNet': UNet}#'CNN': SimpleCNN,,'ResNet': ResNet
for kind in ['RGB', 'SWIR']:

    train_input_img_paths = np.array(images[kind])[train_idxs]
    train_target_class = np.array(wildfire)[train_idxs]
    test_input_img_paths = np.array(images[kind])[mask]
    test_target_class = np.array(wildfire)[mask]
    train_loader = dataset(batch_size=batch_size, img_size=img_size, images_paths=train_input_img_paths, targets=train_target_class, kind=kind)
    test_loader = dataset(batch_size=1, img_size=img_size, images_paths=test_input_img_paths, targets=test_target_class, kind=kind)
    np.savetxt(f'true.txt', test_target_class)
    for m in list(MODELS.keys()):
        if m!='UNet':
            learning_rate = 1e-3
        else:
            learning_rate = 1e-6
        print(f'--------------------Training {m} for {kind}--------------------')
        model = MODELS[m](kind)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_loss, val_loss = [], []
        best_loss = 1
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
            #scheduler.step()

            epoch_train_loss /= len(train_loader)
            train_loss.append(epoch_train_loss)

            model.eval()
            epoch_val_loss = 0.0
            all_targets = []
            all_predictions = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    epoch_val_loss += loss.item()
                    all_targets.extend(targets.cpu().numpy())
                    all_predictions.extend(torch.sigmoid(outputs).cpu().numpy())

            all_predictions = np.array(all_predictions).flatten()
            all_targets = np.array(all_targets).flatten()
            pred_labels = (all_predictions > 0.5).astype(int)

            epoch_val_loss /= len(test_loader)
            val_loss.append(epoch_val_loss)
            accuracy = accuracy_score(test_target_class, pred_labels)*100
            if epoch_val_loss<best_loss:
                torch.save(model.state_dict(), f'./{m}_{kind}.pth')
                np.savetxt(f'./preds/preds_{m}_{kind}.txt', pred_labels)
                best_loss = epoch_val_loss
            print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%, Val Loss: {epoch_val_loss:.4f}, Best Loss: {best_loss:.4f},")

        np.savetxt(f'./{m}_{kind}_train_loss.txt',train_loss)
        np.savetxt(f'./{m}_{kind}_val_loss.txt',val_loss)
