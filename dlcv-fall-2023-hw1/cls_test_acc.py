import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import numpy as np
from p1_dataloader import p1_dataset
import matplotlib.pyplot as plt
from classification_model import VGG13
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    
    
    myseed =  32 # set a random seed for reproducibility
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        
    
    mean = torch.tensor([0.5077, 0.4813, 0.4312])
    std = torch.tensor([0.2000, 0.1986, 0.2034])

    # 32x32 to 8x  256x256
    train_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    valid_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Hyper Parameter
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    BATCH_SIZE = 256

    valid = p1_dataset(root="hw1_data/p1_data/val_50", transform=valid_tfm)
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # for model A
    # model = VGG13()
    # model.load_state_dict(torch.load("model/cls_model/cls/0.657.ckpt"))
    model = torchvision.models.resnet152()
    model.fc = nn.Linear(2048, 50)
    model.load_state_dict(torch.load("model/cls_model/cls/0.8995.ckpt")["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    valid_acc  = list()
    
    with torch.no_grad():
        for data, label in valid_loader:
            
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            pred = model(data)
            pred = torch.argmax(pred, dim=1)
            valid_acc.append(torch.mean((pred == label).float()).item())
        
        mean_valid_acc = sum(valid_acc) / len(valid_acc)
        print(mean_valid_acc)
            

