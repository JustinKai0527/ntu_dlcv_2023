import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as transform
from DANN_dataloader import Mnistm_dataset, SVHN_dataset
from DANN_SVHN_model import FeatureExtractor as FE
from DANN_SVHN_model import LabelPredictor as LP
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __name__ == "__main__":
    
    train_pd = pd.read_csv("hw2_data/digits/svhn/train.csv")
    valid_pd = pd.read_csv("hw2_data/digits/svhn/val.csv")
    
    train_file = train_pd["image_name"].to_numpy()
    train_label = train_pd["label"].to_numpy()
    valid_file = valid_pd["image_name"].to_numpy()
    valid_label = valid_pd["label"].to_numpy()
    
    
    # mnistm tensor([0.4631, 0.4666, 0.4195]) tensor([0.1979, 0.1845, 0.2083])
    # tensor([0.4413, 0.4458, 0.4715]) tensor([0.1169, 0.1206, 0.1042])
    train_tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.4631, 0.4666, 0.4195], std=[0.1979, 0.1845, 0.2083])
    ])
    valid_tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.4413, 0.4458, 0.4715], std=[0.1169, 0.1206, 0.1042])
    ])
    
    # hyper-paramter
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 128
    NUM_EPOCH = 50
    lr = 0.001
    best_acc = 0

    train_data = SVHN_dataset("hw2_data/digits/svhn/data", train_file, train_label, train_tfm)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_data = SVHN_dataset("hw2_data/digits/svhn/data", valid_file, valid_label, valid_tfm)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    
    feature_model = FE().to(DEVICE)
    pred_model = LP().to(DEVICE)

    parameters = list(feature_model.parameters()) + list(pred_model.parameters())
    optimizer = torch.optim.SGD(parameters, lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()    # due to CrossEntropyLoss use softmax first so we don't need softmax in LP
    writer = SummaryWriter()
    print(train_data.__len__(), valid_data.__len__())
    for ep in range(1, NUM_EPOCH+1):
        print("EPOCH:", ep)
        
        feature_model.train(), pred_model.train()
        total_loss = []
        for data, label in tqdm(train_loader):
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            
            pred = pred_model(feature_model(data))
            loss = loss_fn(pred, label)                # loss_fn(pred, target) must
            total_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            
        total_loss = sum(total_loss) / len(total_loss)
        writer.add_scalar("Loss/Train", total_loss, ep)
        
        feature_model.eval(), pred_model.eval()
        with torch.no_grad():
            
            valid_acc = []
            for data, label in valid_loader:
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = pred_model(feature_model(data))
                out = pred.argmax(dim=1)
                
                valid_acc.append((out == label).float().mean().item())
                
            valid_acc = sum(valid_acc) / len(valid_acc)
            print(valid_acc)
            writer.add_scalar("Acc/Valid", valid_acc, ep)
            
            if best_acc < valid_acc:
                best_acc = valid_acc
                print("Save Model")
                
                model_ckpt = {
                    "feature_model": feature_model.state_dict(),
                    "pred_model": pred_model.state_dict()
                }
                
                torch.save(model_ckpt, "modelB.pt")
        
    
    