# -*- coding: utf-8 -*-


! pip install torchmetrics

import pandas as pd
import os
import glob
import cv2
import random
import numpy as np
# from google.colab import drive
import re
import matplotlib.pyplot as plt
from skimage import io

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time
from torchmetrics import Accuracy
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, roc_auc_score, recall_score, jaccard_score

# Data Augmentation
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, ElasticTransform, Normalize
)
from albumentations.pytorch import ToTensorV2

# drive.mount('/content/drive')
base_directory = './lgg-mri-segmentation/kaggle_3m/'
# base_directory = '/content/drive/My Drive/lgg-mri-segmentation/kaggle_3m/'

class MRI_Segmentation:
    def __init__(self, base_directory):
        self.base_directory = base_directory

    def read_csv_data(self):
        return pd.read_csv(self.base_directory + "data.csv")

    def get_data_paths(self):
        data_paths = []
        for sub_dir_path in glob.glob(self.base_directory + "*"):
            try:
                dir_name = sub_dir_path.split('/')[-1]
                for filename in os.listdir(sub_dir_path):
                    mask_path = sub_dir_path + '/' + filename
                    data_paths.extend([dir_name, mask_path])
            except Exception as e:
                print(e)
        return data_paths

    def process_mri_data(self, data_paths):
        filenames = data_paths[::2]
        masks = data_paths[1::2]
        df = pd.DataFrame(data={"patient_id": filenames, "img_path": masks})

        original_img = df[~df['img_path'].str.contains("mask")]
        mask_img = df[df['img_path'].str.contains("mask")]

        imgs = sorted(original_img["img_path"].values, key=lambda x : int(x.split("/")[-1].split("_")[-1][:-4]))
        masks = sorted(mask_img["img_path"].values, key=lambda x : int(x.split("/")[-1].split("_")[-2]))

        return original_img, imgs, masks

    def get_random_image(self, imgs, masks):
        idx = random.randint(0, len(imgs)-1)
        return imgs[idx], masks[idx]

    def resize_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (128, 128))
        return resized_img

    def get_diagnosis(self, img_path):
        image = self.resize_image(img_path)
        value = np.max(image)
        if value > 0:
            return 1
        else:
            return 0

    def generate_mri_df(self, original_img, imgs, masks):
        mri_df = pd.DataFrame({"patient_id": original_img.patient_id.values, "img_path": imgs, 'mask_path': masks})
        mri_df['mask'] = mri_df['mask_path'].apply(lambda x: self.get_diagnosis(x))
        mri_df['mask_path'] = mri_df['mask_path'].apply(lambda x: str(x))
        mri_df.drop(columns=['patient_id'], inplace=True)
        return mri_df

drive = MRI_Segmentation(base_directory)
drive_data = drive.read_csv_data()
data_paths = drive.get_data_paths()
original_img, imgs, masks = drive.process_mri_data(data_paths)
random_img, random_mask = drive.get_random_image(imgs, masks)
mri_df = drive.generate_mri_df(original_img, imgs, masks)
mri_df

mri_df['mask'].value_counts()

mri_df['mask'].value_counts().plot(kind='bar',color=['g','r'],
                title='Count of Tumour vs No Tumour')

#idx = random.randint(0, len(imgs)-1)
count = 0
i = 0
fig,axs = plt.subplots(3,3, figsize=(20,15))
for mask in mri_df['mask']:
    if (mask==1):
      img = io.imread(mri_df.img_path[i])
      print(img.shape)
      axs[count][0].title.set_text("Brain MRI")
      axs[count][0].imshow(img)

      mask = io.imread(mri_df.mask_path[i])
      axs[count][1].title.set_text("Mask =" + str(mri_df['mask'][i]))
      axs[count][1].imshow(mask, cmap='gray')

      img[mask==255] = (255,0,0)  # change pixel color at the position of mask
      axs[count][2].title.set_text("MRI with Mask =" + str(mri_df['mask'][i]))
      axs[count][2].imshow(img)
      count +=1
    i += 1
    if (count==3):
        break

fig.tight_layout()

count = 0
i = 0
fig,axs = plt.subplots(3,3, figsize=(20,15))
for mask in mri_df['mask']:
    if (mask==0):
      img = io.imread(mri_df.img_path[i])
      #print(img.shape)
      axs[count][0].title.set_text("Brain MRI")
      axs[count][0].imshow(img)

      mask = io.imread(mri_df.mask_path[i])
      axs[count][1].title.set_text("Mask =" + str(mri_df['mask'][i]))
      axs[count][1].imshow(mask, cmap='gray')

      img[mask==255] = (255,0,0)  # change pixel color at the position of mask
      axs[count][2].title.set_text("MRI with Mask =" + str(mri_df['mask'][i]))
      axs[count][2].imshow(img)
      count +=1
    i += 1
    if (count==3):
        break

fig.tight_layout()

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_1 = double_conv(in_channels, 32)
        self.down_conv_2 = double_conv(32, 64)
        self.down_conv_3 = double_conv(64, 128)
        self.down_conv_4 = double_conv(128, 256)
        self.down_conv_5 = double_conv(256, 512)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(512, 256)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(256, 128)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(128, 64)
        self.up_trans_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(64, 32)

        self.out = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    def forward(self, image):
        # Encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        # Decoder
        x = self.up_trans_1(x9)
        x = self.up_conv_1(torch.cat([x, x7], 1))

        x = self.up_trans_2(x)
        x = self.up_conv_2(torch.cat([x, x5], 1))

        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x, x3], 1))

        x = self.up_trans_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))

        x = self.out(x)
        return x

model = UNet(in_channels=3, out_channels=1)

class MRIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        self.augmentations = Compose([
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            ElasticTransform(p=0.2),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['img_path']
        mask_path = self.data.iloc[idx]['mask_path']

        image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 128))
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (128, 128))

        augmented = self.augmentations(image=image, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        image = image / 255.0
        mask = mask / 255.0
        # image = image.transpose((2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        if image.shape[0] != 3:
            image = np.repeat(image, 3, axis=0)
        return {'image': image, 'mask': mask}

dataset = MRIDataset(dataframe=mri_df, transform=None)
plt.figure(figsize=(12, 12))
for i in range(25):
    sample = dataset[i]
    image = sample['image'].permute(1,2,0).cpu().numpy()
    #mask = sample['mask'].numpy()
    mask = sample['mask']

    plt.subplot(5, 5, i+1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Image')

plt.tight_layout()
plt.show()

num_samples = len(mri_df)
num_train = int(num_samples * 0.8)
num_val = int(num_samples * 0.1)
num_test = num_samples - num_train - num_val

train_data = mri_df[:num_train]
val_data = mri_df[num_train:num_train+num_val]
test_data = mri_df[num_train+num_val:]

train_dataset = MRIDataset(train_data)
val_dataset = MRIDataset(val_data)
test_dataset = MRIDataset(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

##criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
device

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    return (2 * intersection + smooth) / (union + smooth)

def iou_coef(y_true, y_pred, smooth=1e-6):
    intersection = torch.sum(y_true * y_pred)
    sum_vals = torch.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum_vals - intersection + smooth)
    return iou
class DiceCoefficient:
    def __init__(self):
        self.inter = 0
        self.union = 0

    def update(self, y_true, y_pred):
        self.inter += torch.sum(y_true * y_pred)
        self.union += torch.sum(y_true) + torch.sum(y_pred)

    def compute(self, smooth=1e-6):
        return (2 * self.inter + smooth) / (self.union + smooth)

    def reset(self):
        self.inter = 0
        self.union = 0

class IoUCoefficient:
    def __init__(self):
        self.inter = 0
        self.sum_vals = 0

    def update(self, y_true, y_pred):
        self.inter += torch.sum(y_true * y_pred)
        self.sum_vals += torch.sum(y_true + y_pred)

    def compute(self, smooth=1e-6):
        return (self.inter + smooth) / (self.sum_vals - self.inter + smooth)

    def reset(self):
        self.inter = 0
        self.sum_vals = 0

def train(model, device, train_loader, optimizer, criterion, epoch, dice_coeff, iou_coeff):
    model.train()
    loss_train = AverageMeter()
    num_classes = 2
    acc_train = Accuracy(num_classes=num_classes, task="BINARY").to(device)
    with tqdm(train_loader, unit='batch') as tepoch:
        for batch in tepoch:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, masks)

            dice_coeff_train = dice_coef(masks, output)
            iou_coeff_train = iou_coef(masks, output)
            ##dice_coeff.update(dice_coeff_train)
            ##iou_coeff.update(iou_coeff_train)
            dice_coeff.update(masks, output)
            iou_coeff.update(masks, output)

            loss.backward()
            optimizer.step()

            loss_train.update(loss.item())
            acc_train(output, masks.int())

            tepoch.set_postfix(loss=loss_train.avg,
                               accuracy=100.*acc_train.compute().item(),
                               dice_coeff=dice_coeff_train.item(),
                               iou_score=iou_coeff_train.item())

    return model, loss_train.avg, acc_train.compute().item()

def validate(model, device, valid_loader, loss_fn, dice_coeff, iou_coeff):
    model.eval()
    num_classes = 2
    with torch.no_grad():
        loss_valid = AverageMeter()
        acc_valid = Accuracy(num_classes=num_classes, task="BINARY").to(device)
        for i, batch in enumerate(valid_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            dice_coeff_val = dice_coef(masks, outputs)
            iou_coeff_val = iou_coef(masks, outputs)
            dice_coeff.update(masks, outputs)
            iou_coeff.update(masks, outputs)
            ##dice_coeff.update(dice_coeff_val)
            ##iou_coeff.update(iou_coeff_val)

            loss_valid.update(loss.item())
            acc_valid(outputs, masks.int())

    return loss_valid.avg, acc_valid.compute().item()

def evaluate(model, device, test_loader, loss_fn, dice_coeff, iou_coeff):
    model.eval()
    num_classes = 2
    with torch.no_grad():
        loss_valid = AverageMeter()
        acc_valid = Accuracy(num_classes=num_classes, task="BINARY").to(device)
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            dice_coeff_val = dice_coef(masks, outputs)
            iou_coeff_val = iou_coef(masks, outputs)
            ##dice_coeff.update(dice_coeff_val)
            ##iou_coeff.update(iou_coeff_val)
            dice_coeff.update(masks, outputs)
            iou_coeff.update(masks, outputs)

            loss_valid.update(loss.item())
            acc_valid(outputs, masks.int())

    return loss_valid.avg, acc_valid.compute().item()

def train_and_validate_evaluate_model(model, optimizer, epochs, device, train_loader, valid_loader, test_loader, loss_fn):
    loss_train_hist = []
    loss_test_hist = []
    loss_valid_hist = []
    acc_valid_hist = []
    acc_train_hist = []
    acc_test_hist = []
    dice_coeff_train_hist = []
    dice_coeff_valid_hist = []
    dice_coeff_test_hist = []
    iou_score_train_hist = []
    iou_score_valid_hist = []
    iou_score_test_hist = []

    dice_coeff_train = DiceCoefficient()
    iou_coeff_train = IoUCoefficient()

    dice_coeff_valid = DiceCoefficient()
    iou_coeff_valid = IoUCoefficient()

    dice_coeff_test = DiceCoefficient()
    iou_coeff_test = IoUCoefficient()

    for epoch in range(1, epochs + 1):
        model, loss_train, acc_train = train(model, device, train_loader, optimizer, loss_fn, epoch, dice_coeff_train, iou_coeff_train)
        loss_valid, acc_valid = validate(model, device, valid_loader, loss_fn, dice_coeff_valid, iou_coeff_valid)
        loss_test, acc_test = evaluate(model, device, test_loader, loss_fn, dice_coeff_test, iou_coeff_test)

        dice_coeff_train_val = dice_coeff_train.compute().item()
        iou_score_train_val = iou_coeff_train.compute().item()
        dice_coeff_valid_val = dice_coeff_valid.compute().item()
        iou_score_valid_val = iou_coeff_valid.compute().item()
        dice_coeff_test_val = dice_coeff_test.compute().item()
        iou_score_test_val = iou_coeff_test.compute().item()

        loss_train_hist.append(loss_train)
        loss_valid_hist.append(loss_valid)
        loss_test_hist.append(loss_test)
        acc_train_hist.append(acc_train)
        acc_valid_hist.append(acc_valid)
        acc_test_hist.append(acc_test)
        dice_coeff_train_hist.append(dice_coeff_train_val)
        dice_coeff_valid_hist.append(dice_coeff_valid_val)
        dice_coeff_test_hist.append(dice_coeff_test_val)

        iou_score_train_hist.append(iou_score_train_val)
        iou_score_valid_hist.append(iou_score_valid_val)
        iou_score_test_hist.append(iou_score_test_val)

        print(f'Epoch {epoch}, Valid: Loss={loss_valid:.4f}, Accuracy={acc_valid:.4f}, Dice Coef={dice_coeff_valid_val:.4f}, IoU Score={iou_score_valid_val:.4f}')
        print(f'Epoch {epoch}, Test: Loss={loss_test:.4f}, Accuracy={acc_test:.4f}, Dice Coef={dice_coeff_test_val:.4f}, IoU Score={iou_score_test_val:.4f}\n')

    return loss_train_hist, loss_valid_hist, loss_test_hist, acc_train_hist, acc_valid_hist, acc_test_hist, dice_coeff_train_hist, dice_coeff_valid_hist, dice_coeff_test_hist, iou_score_train_hist,iou_score_valid_hist,iou_score_test_hist

epochs = 20
loss_train, loss_valid, loss_test, acc_train, acc_valid, acc_test, dice_coef_train,dice_coef_valid ,dice_coef_test, iou_score_train, iou_score_valid, iou_score_test = train_and_validate_evaluate_model(model, optimizer, epochs, device, train_dataloader, val_dataloader, test_dataloader, criterion)

def plot_metrics(loss_train_hist, loss_valid_hist, loss_test_hist,acc_train_hist,acc_valid_hist, acc_test_hist,dice_coeff_train,dice_coeff_valid,dice_coeff_test,iou_score_train,iou_score_valid,iou_score_test):
    epochs = range(1, len(loss_train_hist) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss_train_hist, label='Training Loss')
    plt.plot(epochs, loss_valid_hist, label='Validation Loss')
    plt.plot(epochs, loss_test_hist, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, acc_train_hist, label='Training Accuracy')
    plt.plot(epochs, acc_valid_hist, label='Validation Accuracy')
    plt.plot(epochs, acc_test_hist, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, dice_coeff_train, label='Training Dice Coefficient')
    plt.plot(epochs, dice_coeff_valid, label='Validation Dice Coefficient')
    plt.plot(epochs, dice_coeff_test, label='Testing Dice Coefficient')

    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, iou_score_train, label='Training IoU Score')
    plt.plot(epochs, iou_score_valid, label='Validation IoU Score')
    plt.plot(epochs, iou_score_test, label='Testing IoU Score')

    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(loss_train, loss_valid, loss_test, acc_train, acc_valid, acc_test, dice_coef_train,dice_coef_valid,dice_coef_test, iou_score_train,iou_score_valid,iou_score_test )

batch = next(iter(test_dataloader))
with torch.no_grad():
    model.eval()
    logits = model(batch['image'].to(device))
    pr_masks = (logits.squeeze(1) > 0.5).float()

for image, gt_mask, pr_mask in zip(batch['image'], batch['mask'], pr_masks):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.numpy().squeeze(), cmap = 'gray') # just squeeze classes dim, because we have only one class
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pr_mask.detach().cpu().numpy()) # just squeeze classes dim, because we have only one class
    plt.title("Prediction")
    plt.axis("off")

    plt.show()

