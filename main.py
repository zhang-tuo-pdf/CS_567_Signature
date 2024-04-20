import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, random_split
from data.snn_dataset import SiameseDataset
from model.snn import SiameseNetwork, ContrastiveLoss
import torch.nn.functional as F
import numpy as np

def data_loading(batchsize):
    training_dir = '/kaggle/input/sign-data/sign_data/train'
    testing_dir = '/kaggle/input/sign-data/sign_data/test'
    training_csv = '/kaggle/input/sign-data/sign_data/train_data.csv'
    testing_csv = '/kaggle/input/sign-data/sign_data/test_data.csv'

    transform = ToTensor()

    siamese_dataset = SiameseDataset(
        training_csv=training_csv,
        training_dir=training_dir,
        transform=transform
    )

    train_size = int(0.8 * len(siamese_dataset))
    val_size = len(siamese_dataset) - train_size
    train_dataset, val_dataset = random_split(siamese_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=batchsize)

    val_dataloader = DataLoader(val_dataset,
                                shuffle=True,
                                num_workers=8,
                                batch_size=1)

    test_dataset = SiameseDataset(
        training_csv=testing_csv,
        training_dir=testing_dir,
        transform=transform
    )

    test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=1, shuffle=True)

    # Print the size of datasets
    print('size of trainset is ', len(train_dataset))
    print('size of valset is ', len(val_dataset))
    print('size of testset is ', len(test_dataset))

    # Return the DataLoaders
    return train_dataloader, val_dataloader, test_dataloader



def load_model():
    model = SiameseNetwork()
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
    train_dataloader, validation_dataloader, test_dataloader = data_loading(batch_size)
    snn = load_model()
    device = 'cpu'
    learning_rate = 5e-4
    training_epoch = 10
    snn, epoch_loss = train(train_dataloader, snn, device, learning_rate, training_epoch)
    #torch.save(snn.state_dict(), 'snn_trained_2_64.pth')
    #snn.load_state_dict(torch.load("snn_trained_2.pth"))
    threshold = get_threshold(snn, device, validation_dataloader)
    acc = test(snn, device, test_dataloader, threshold)
    print(acc)
