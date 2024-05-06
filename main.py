import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose
from data.icdar_dataset import SiameseDataset_ICDAR
from data.msds_dataset import SiameseDataset_MSDS
from model.snn import SiameseNetwork, ContrastiveLoss


def data_loading(batchsize, dataset):
    if dataset == 'icdar':
        # change paths
        training_dir = '/kaggle/input/sign-data/sign_data/train'
        testing_dir = '/kaggle/input/sign-data/sign_data/test'
        training_csv = '/kaggle/input/sign-data/sign_data/train_data.csv'
        testing_csv = '/kaggle/input/sign-data/sign_data/test_data.csv'

        transform = ToTensor()

        siamese_dataset = SiameseDataset_ICDAR(training_csv=training_csv, training_dir=training_dir, transform=transform)
        test_dataset = SiameseDataset_ICDAR(training_csv=testing_csv, training_dir=testing_dir, transform=transform)

        train_size = int(0.8 * len(siamese_dataset))
        val_size = len(siamese_dataset) - train_size
        train_dataset, val_dataset = random_split(siamese_dataset, [train_size, val_size])

        # create dataLoaders
        train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=batchsize)

        val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=4, batch_size=1)

        test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1, shuffle=True)

    elif dataset == 'msds':
        # change paths
        training_dir = 'data/MSDS/train'
        testing_dir = 'data/MSDS/test'
        val_dir = 'data/MSDS/val'
        training_csv = 'data/MSDS/train_data.csv'
        testing_csv = 'data/MSDS/test_data.csv'
        val_csv = 'data/MSDS/val_data.csv'

        transform = ToTensor()
        
        train_dataset_full = SiameseDataset_MSDS(training_csv=training_csv, training_dir=training_dir, transform=transform)
        val_dataset_full = SiameseDataset_MSDS(training_csv=val_csv, training_dir=val_dir, transform=transform)
        test_dataset_full = SiameseDataset_MSDS(training_csv=testing_csv, training_dir=testing_dir, transform=transform)

        # randomly pick 1/3 of full dataset
        train_indices = np.random.choice(len(train_dataset_full), len(train_dataset_full) // 3, replace=False)
        train_dataset = Subset(train_dataset_full, train_indices)

        val_indices = np.random.choice(len(val_dataset_full), len(val_dataset_full) // 3, replace=False)
        val_dataset = Subset(val_dataset_full, val_indices)

        test_indices = np.random.choice(len(test_dataset_full), len(test_dataset_full) // 3, replace=False)
        test_dataset = Subset(test_dataset_full, test_indices)
        
        # create dataLoaders
        train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=batchsize, prefetch_factor=2)
        val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=4, batch_size=1, prefetch_factor=2)
        test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1, shuffle=True, prefetch_factor=2)

    # check if empty
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Check dataset paths and CSV file.")
    
    # print sizes of datasets
    print('size of trainset is ', len(train_dataset))
    print('size of valset is ', len(val_dataset))
    print('size of testset is ', len(test_dataset))

    return train_dataloader, val_dataloader, test_dataloader


def load_model(encoder='cnn'):
    model = SiameseNetwork(model=encoder)
    return model

def train(train_data, model, device, learning_rate, training_epoch):
    model.to(device)
    model.train()
    criterion = ContrastiveLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    epoch_loss = []
    print("training_epoch is ", training_epoch)
    print("training_epoch number is ", range(training_epoch))
    for epoch in range(training_epoch):
        batch_loss = []
        for batch_idx, data in enumerate(train_data, 0):
            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            output1,output2 = model(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            batch_loss.append(loss_contrastive.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print("Epoch number {}\n Current loss {}\n".format(epoch,epoch_loss[epoch]))
    return model, epoch_loss

def get_threshold(model, device, val_dataloader):
    model.to(device)
    model.eval()
    forge_distance = []
    true_distance = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader, 0):
            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device), label.to(device)
            output1,output2 = model(img0,img1)
            EuDist = F.pairwise_distance(output1, output2)
            if label == 1:
                forge_distance.append(EuDist.detach().cpu().numpy())
            if label == 0:
                true_distance.append(EuDist.detach().cpu().numpy())
    threshold = np.mean(true_distance) + np.std(true_distance)
    return threshold

def test(model, device, test_data, threshold):
    model.to(device)
    model.eval()
    with torch.no_grad():
        pred = 0
        correct = 0
        count = 0
        count_1 = 0
        for batch_idx, data in enumerate(test_data, 0):
            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device), label.to(device)
            output1,output2 = model(img0,img1)
            EuDist = F.pairwise_distance(output1, output2)
            if EuDist <= threshold:
                pred = 0
            else:
                pred = 1
            if pred == label.detach().cpu().numpy():
                correct += 1
            count += 1
    acc = correct / count
    return acc


if __name__ == "__main__":
    batch_size = 16
    dataset = 'icdar'
    train_dataloader, validation_dataloader, test_dataloader = data_loading(batch_size, dataset)
    snn = load_model(encoder = 'cnn')
    device = 'cuda'
    learning_rate = 5e-4
    training_epoch = 10
    snn, epoch_loss = train(train_dataloader, snn, device, learning_rate, training_epoch)
    #torch.save(snn.state_dict(), 'snn_trained_2_64.pth')
    #snn.load_state_dict(torch.load("snn_trained_2.pth"))
    threshold = get_threshold(snn, device, validation_dataloader)
    acc = test(snn, device, test_dataloader, threshold)
    print(acc)
