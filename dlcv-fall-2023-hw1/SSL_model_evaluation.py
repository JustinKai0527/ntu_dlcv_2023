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
    train_set = OfficeDataset("hw1_data/p2_data/office/train", transform=train_tfm)
    valid_set = OfficeDataset("hw1_data/p2_data/office/val", transform=valid_tfm)
    
    # print(train_set.__len__(), valid_set.__len__())      # (3951 406
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=4)
    
    
    backbone = torchvision.models.resnet50()
    
    model = nn.Sequential(
        backbone,
        nn.Dropout(0.25),
        nn.Linear(1000, 65)
    )

    # hyper-parameter
    writer = SummaryWriter()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCH = 150
    lr = 5e-4
    best_acc = 0
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr)


    for ep in range(1, NUM_EPOCH+1):
        print(f"Epoch {ep}")
        model.train()
        train_loss = []
        train_acc = []
        
        for data, label in tqdm(train_loader):
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            
            pred = model(data)
            loss = loss_fn(pred, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            pred = torch.argmax(pred, dim=1)
            train_acc.append(torch.mean((pred == label).float()).item())
        
        print(sum(train_loss) / len(train_loss))
        print(sum(train_acc) / len(train_acc))
        writer.add_scalar("Loss/Train", sum(train_loss) / len(train_loss), ep)
        writer.add_scalar("acc/Train", sum(train_acc) / len(train_acc), ep)
        
        valid_loss = []
        valid_acc = []
        
        with torch.no_grad():
            
            model.eval()
            for data, label in valid_loader:
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = model(data)
                loss = loss_fn(pred, label)
                
                valid_loss.append(loss.item())
                pred = torch.argmax(pred, dim=1)
                valid_acc.append(torch.mean((pred == label).float()).item())
                
            print(sum(valid_loss) / len(valid_loss))
            writer.add_scalar("Loss/Valid", sum(valid_loss) / len(valid_loss), ep)
            writer.add_scalar("acc/Valid", sum(valid_acc) / len(valid_acc), ep)
            
            valid_acc = sum(valid_acc) / len(valid_acc)
            print(valid_acc)
            
            if valid_acc >= best_acc:
                best_acc = valid_acc
                print("Model Saved")
                torch.save(model.state_dict(), "model_A.pt")
        
        