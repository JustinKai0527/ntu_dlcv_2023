import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as transform
from DANN_dataloader import USPS_dataset
from DANN_USPS_model import FeatureExtractor as FE
from DANN_USPS_model import LabelPredictor as LP
import pandas as pd

if __name__ == "__main__":
    
    valid_pd = pd.read_csv("hw2_data/digits/usps/val.csv")
    
    valid_file = valid_pd["image_name"].to_numpy()
    valid_label = valid_pd["label"].to_numpy()
    
    valid_tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.2570, 0.2570, 0.2570], std=[0.3372, 0.3372, 0.3372]),
    ])
    
    dataset = USPS_dataset("hw2_data/digits/usps/data", valid_file, valid_label, valid_tfm)
    # print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_model = FE().to(DEVICE)
    pred_model = LP().to(DEVICE)
    
    model_ckpt = torch.load("modelE.pt")
    feature_model.load_state_dict(model_ckpt["feature_model"])
    pred_model.load_state_dict(model_ckpt["label_pred_model"])
    
    feature_model.eval(), pred_model.eval()
    print(dataset.__len__())
    with torch.no_grad():
        total_acc = []
        for data, label in dataloader:
            print(data.shape)
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            
            pred = pred_model(feature_model(data))
            out = pred.argmax(dim=1)
            
            total_acc.append((out == label).float().mean().item())
        
        total_acc = sum(total_acc) / len(total_acc)
        print(total_acc)
        
        
          
                        
#                         MNIST-M → SVHN   |   MNIST-M → USPS
# Trained on source       0.28353(0.957)   |       0.6708(0.9355)
#                                          |
# Adaptation (DANN)                        |
#                                          |
# Trained on target          0.91405       |       0.9558
 
 
