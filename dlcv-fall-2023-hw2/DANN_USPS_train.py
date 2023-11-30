import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
from DANN_dataloader import Mnistm_dataset, USPS_dataset
from DANN_USPS_model import FeatureExtractor as FE
from DANN_USPS_model import LabelPredictor as LP
from DANN_USPS_model import DomainClassifier as DC
import numpy as np
import pandas as pd
from tqdm import tqdm


class RepeatDataset(Dataset):
    def __init__(self, original_dataset, repeat_factor=10):
        self.original_dataset = original_dataset
        self.repeat_factor = repeat_factor
        self.total_len = len(original_dataset) * repeat_factor

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        original_index = index % len(self.original_dataset)
        return self.original_dataset[original_index]
    
    
if __name__ == "__main__":
    
    train_source_pd = pd.read_csv("hw2_data/digits/mnistm/train.csv")
    train_target_pd = pd.read_csv("hw2_data/digits/usps/train.csv")
    valid_pd = pd.read_csv("hw2_data/digits/usps/val.csv")
    
    train_source_file = train_source_pd["image_name"].to_numpy()
    train_source_label = train_source_pd["label"].to_numpy()
    train_target_file = train_target_pd["image_name"].to_numpy()
    train_target_label = train_target_pd["label"].to_numpy()
    valid_file = valid_pd["image_name"].to_numpy()
    valid_label = valid_pd["label"].to_numpy()
    
    # mnistm tensor([0.4631, 0.4666, 0.4195]) tensor([0.1979, 0.1845, 0.2083])
    # svhn tensor([0.4413, 0.4458, 0.4715]) tensor([0.1169, 0.1206, 0.1042])
    # usps tensor([0.2570, 0.2570, 0.2570]) tensor([0.3372, 0.3372, 0.3372])
    train_source_tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.4631, 0.4666, 0.4195], std=[0.1979, 0.1845, 0.2083])
    ])
    train_target_tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.2570, 0.2570, 0.2570], std=[0.3372, 0.3372, 0.3372])
    ])
    valid_tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.2570, 0.2570, 0.2570], std=[0.3372, 0.3372, 0.3372])
    ])
    
    train_source_data = Mnistm_dataset("hw2_data/digits/mnistm/data", train_source_file, train_source_label, train_source_tfm, domain=True)
    train_target_data = USPS_dataset("hw2_data/digits/usps/data", train_target_file, train_target_label, train_target_tfm, domain=True)
    valid_data = USPS_dataset("hw2_data/digits/usps/data", valid_file, valid_label, valid_tfm, domain=True)
    
    train_target_data = RepeatDataset(train_target_data, repeat_factor=10)
    
    print(train_source_data.__len__(), train_target_data.__len__(), valid_data.__len__())
    
    # hyper-parameter
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 128
    NUM_EPOCH = 100
    gamma = 10       # default
    lr = 0.001
    best_acc = 0
    
    train_source_loader = DataLoader(train_source_data, batch_size=BATCH_SIZE, shuffle=True)
    train_target_loader = DataLoader(train_target_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    
    feature_model = FE().to(DEVICE)
    label_pred_model = LP().to(DEVICE)
    domain_pred_model = DC().to(DEVICE)
    
    parameters = list(feature_model.parameters()) + list(label_pred_model.parameters()) + list(domain_pred_model.parameters())
    optimizer = torch.optim.SGD(parameters, lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.7, total_iters=10)
    label_loss_fn = nn.CrossEntropyLoss()
    domain_loss_fn = nn.BCELoss()

    for ep in range(1, NUM_EPOCH+1):
        
        print("Epoch:", ep)
        total_loss = []
        feature_model.train(), label_pred_model.train(), domain_pred_model.train()
        # the source_loader and target_loader is not same len the zip will zip the min len and ignore the longer one data
        for (data_s, domain_s, label_s), (data_t, domain_t, _) in tqdm(zip(train_source_loader, train_target_loader)):
            
            data_s = data_s.to(DEVICE)
            data_t = data_t.to(DEVICE)
            label_s = label_s.to(DEVICE)
            Lambda = (2 / (1 + np.exp(- gamma * (ep / NUM_EPOCH)))) - 1
            optimizer.zero_grad()
            
            # compute LCE
            feature_s = feature_model(data_s)
            label_pred = label_pred_model(feature_s)
            LCE = label_loss_fn(label_pred, label_s).to(DEVICE)                # loss_fn(pred, target) necessary
            
            # compute LBCE
            feature_t = feature_model(data_t)
            
            pred_domain_s = domain_pred_model(feature_s, Lambda).squeeze()
            pred_domain_t = domain_pred_model(feature_t, Lambda).squeeze()
            domain_s = domain_s.float().to(DEVICE)
            domain_t = domain_t.float().to(DEVICE)
            
            LBCE = domain_loss_fn(pred_domain_s, domain_s).to(DEVICE) + domain_loss_fn(pred_domain_t, domain_t).to(DEVICE)
            
            # compute final loss = LCE + LBCE
            loss = LCE + LBCE

            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            
        scheduler.step()
        
        total_loss = sum(total_loss) / len(total_loss)
        print(f"Loss: {total_loss}")
        
        feature_model.eval(), label_pred_model.eval(), domain_pred_model.eval()
        with torch.no_grad():
            total_acc = []
            for data, _, label in valid_loader:
                # print(data.shape)
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = label_pred_model(feature_model(data))
                out = pred.argmax(dim=1)
                
                total_acc.append((out == label).float().mean().item())
            
            total_acc = sum(total_acc) / len(total_acc)
            print(total_acc)

            if best_acc < total_acc:
                print("Save model")
                best_acc = total_acc
                model_ckpt = {
                    "feature_model": feature_model.state_dict(),
                    "label_pred_model": label_pred_model.state_dict(),
                }
                
                torch.save(model_ckpt, "modelE.pt")