import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as transform
from DANN_dataloader import SVHN_dataset
from DANN_SVHN_model import FeatureExtractor as FE
from DANN_SVHN_model import LabelPredictor as LP
import pandas as pd

if __name__ == "__main__":
    
    valid_pd = pd.read_csv("hw2_data/digits/svhn/val.csv")
    
    valid_file = valid_pd["image_name"].to_numpy()
    valid_label = valid_pd["label"].to_numpy()
    
    valid_tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.4413, 0.4458, 0.4715], std=[0.1169, 0.1206, 0.1042])
    ])
    
    dataset = SVHN_dataset("hw2_data/digits/svhn/data", valid_file, valid_label, valid_tfm)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_model = FE().to(DEVICE)
    pred_model = LP().to(DEVICE)
    
    model_ckpt = torch.load("DANN_model/modelB.pt")
    feature_model.load_state_dict(model_ckpt["feature_model"])
    pred_model.load_state_dict(model_ckpt["label_pred_model"])
    
    feature_model.eval(), pred_model.eval()
    
    with torch.no_grad():
        total_acc = []
        for data, label in dataloader:
            
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            
            pred = pred_model(feature_model(data))
            out = pred.argmax(dim=1)
            
            total_acc.append((out == label).float().mean().item())
        
        total_acc = sum(total_acc) / len(total_acc)
        print(total_acc)
        
        
        
          
                        
#                         MNIST-M → SVHN   |   MNIST-M → USPS
# Trained on source       0.28353(0.957)   |
#                                          |
# Adaptation (DANN)                        |
#                                          |
# Trained on target          0.91405       |
 
 
