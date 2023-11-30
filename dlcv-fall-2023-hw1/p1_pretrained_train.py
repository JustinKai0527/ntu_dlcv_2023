import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import numpy as np
from p1_dataloader import p1_dataset
import timm
from timm.loss import SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from tqdm import tqdm
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
    BATCH_SIZE = 32
    lr = 0.003
    NUM_EPOCH = 20

    train = p1_dataset(root="hw1_data/p1_data/train_50", transform=train_tfm)
    valid = p1_dataset(root="hw1_data/p1_data/val_50", transform=valid_tfm)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # print(timm.list_models('resnet*'))
    # find this ViT model in hugginface https://huggingface.co/timm/vit_srelpos_medium_patch16_224.sw_in1k/discussions?status=all
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 50)
    model = model.to(DEVICE)
    
    mixup = Mixup(
    mixup_alpha=0.1,
    cutmix_alpha=1,
    cutmix_minmax=None,
    prob=1,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=50,
    )   
    
    optimizer = optim.SGD(model.parameters(), lr)
    criterion = SoftTargetCrossEntropy()
    writer = SummaryWriter()
    # model.load_state_dict(torch.load("best_model.ckpt")["model_state_dict"])
    for ep in range(1, NUM_EPOCH+1):
        
        print(f"Epoch {ep}")
        train_loss = list()
        train_acc = list()
        model.train()
        
        for batch_idx, (data, label) in tqdm(enumerate(train_loader)):
            
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            data, label = mixup(data, label)
            optimizer.zero_grad()
            
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        
            # print(pred.shape)
            train_loss.append(loss.item())
        
        
        mean_train_loss = sum(train_loss) / len(train_loss)
        print("Loss", mean_train_loss)
        writer.add_scalar("Loss/Train", mean_train_loss, ep)
        
        model.eval()
        valid_loss = list()
        valid_acc  = list()
        
        with torch.no_grad():
            for data, label in valid_loader:
                
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                pred = model(data)
                valid_loss.append(nn.CrossEntropyLoss()(pred, label).item())
                pred = torch.argmax(pred, dim=1)
                valid_acc.append(torch.mean((pred == label).float()).item())
            
            mean_valid_loss = sum(valid_loss) / len(valid_loss)
            mean_valid_acc = sum(valid_acc) / len(valid_acc)
            writer.add_scalar("Loss/Valid", mean_valid_loss, ep)
            writer.add_scalar("Acc/Valid", mean_valid_acc, ep)
            # print(mean_valid_acc)
            if best_acc <= mean_valid_acc:
                print("Save Model")
                best_acc = mean_valid_acc
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, "cls/best_model.ckpt")
                print(best_acc)
