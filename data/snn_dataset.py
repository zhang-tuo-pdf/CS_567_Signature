import os
import cv2
import numpy as np
from PIL import Image, ImageChops
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else None

def process_image(image, output_size=(105, 105)):
    img_trimmed = trim(image)

    img_padded = cv2.copyMakeBorder(
        np.array(img_trimmed),
        top=350, bottom=350, left=550, right=550,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    center_y, center_x = img_padded.shape[:2]
    crop_img_final = img_padded[
        max(center_y//2 - output_size[0]//2, 0):min(center_y//2 + output_size[0]//2, center_y),
        max(center_x//2 - output_size[1]//2, 0):min(center_x//2 + output_size[1]//2, center_x)
    ]
    return Image.fromarray(cv2.resize(crop_img_final, output_size, interpolation=cv2.INTER_AREA), 'L')

class SiameseDataset(Dataset):
    def __init__(self, training_csv, training_dir, transform=None):
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform
        self.image_paths = [
            (os.path.join(self.train_dir, row['image1']),
             os.path.join(self.train_dir, row['image2']))
            for index, row in self.train_df.iterrows()
        ]
        self.labels = self.train_df['label'].astype(int).values

    def __getitem__(self, index):
        image1_path, image2_path = self.image_paths[index]
        
        img0 = Image.open(image1_path).convert("L")
        img1 = Image.open(image2_path).convert("L")

        img0 = process_image(img0)
        img1 = process_image(img1)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return img0, img1, label

    def __len__(self):
        return len(self.image_paths)

