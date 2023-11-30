import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from SSL_dataloader import OfficeDataset
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    
    train_tfm = transforms.Compose([
        transforms.RandomApply(
            [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
            p = 0.3
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.GaussianBlur((3, 3), (1.0, 2.0))],
            p = 0.2
        ),
        transforms.RandomResizedCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.6062, 0.5748, 0.5421]),
            std=torch.tensor([0.2397, 0.2408, 0.2455])),
    ])
    
    valid_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.6062, 0.5748, 0.5421]),
            std=torch.tensor([0.2397, 0.2408, 0.2455])),
    ])
    
    # loading data
    valid_set = OfficeDataset("hw1_data/p2_data/office/val", transform=valid_tfm)
    
    # print(train_set.__len__(), valid_set.__len__())      # (3951 406
    valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=4)
    
    
    backbone = torchvision.models.resnet50()
    
    model = nn.Sequential(
        backbone,
        nn.Dropout(0.25),
        nn.Linear(1000, 65)
    )
    
    model.load_state_dict(torch.load("model/SSL_model/best_model.pt"))
    #hyper-paramter
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(DEVICE)

    valid_acc = []
    
    with torch.no_grad():
        
        model.eval()
        for data, label in valid_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            
            pred = model(data)

            pred = torch.argmax(pred, dim=1)
            valid_acc.append(torch.mean((pred == label).float()).item())
            

        valid_acc = sum(valid_acc) / len(valid_acc)
        print(valid_acc)
        
        