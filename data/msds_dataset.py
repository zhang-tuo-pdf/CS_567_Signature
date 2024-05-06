import os
import cv2
import numpy as np
from PIL import Image, ImageChops
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def trim_msds(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im

def resize_image_keep_aspect(image, max_dimension=105):
    original_width, original_height = image.size
    ratio = min(max_dimension / original_width, max_dimension / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

def process_image_msds(image, max_dimension=105):
    img_trimmed = trim_msds(image)

    img_resized = resize_image_keep_aspect(img_trimmed, max_dimension)

    img_padded = Image.new('L', (max_dimension, max_dimension), (255))
    img_padded.paste(img_resized, ((max_dimension - img_resized.width) // 2, (max_dimension - img_resized.height) // 2))

    return img_padded

class SiameseDataset_MSDS(Dataset):
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

    def remove_transparency(self, image, bg_color=(255, 255, 255)):
        # image is RGBA; A for transparency
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            alpha = image.convert('RGBA').split()[-1]
            bg = Image.new("RGBA", image.size, bg_color + (255,))
            bg.paste(image, mask=alpha)
            return bg.convert('RGB')
        else:
            return image

    def __getitem__(self, index):
        image1_path, image2_path = self.image_paths[index]

        img0 = Image.open(image1_path)
        img0 = self.remove_transparency(img0)
        img0 = img0.convert('L')

        img1 = Image.open(image2_path)
        img1 = self.remove_transparency(img1)
        img1 = img1.convert('L')

        img0 = process_image_msds(img0)
        img1 = process_image_msds(img1)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return img0, img1, label

    def __len__(self):
        return len(self.image_paths)