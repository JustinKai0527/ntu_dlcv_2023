import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.transforms as transform
from DANN_dataloader import USPS_dataset, Mnistm_dataset
from DANN_USPS_model import FeatureExtractor as FE
from DANN_USPS_model import LabelPredictor as LP
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    train_pd = pd.read_csv("hw2_data/digits/mnistm/val.csv")
    valid_pd = pd.read_csv("hw2_data/digits/usps/val.csv")
    
    train_file = train_pd["image_name"].to_numpy()
    train_label = train_pd["label"].to_numpy()
    valid_file = valid_pd["image_name"].to_numpy()
    valid_label = valid_pd["label"].to_numpy()
    
    train_tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.4631, 0.4666, 0.4195], std=[0.1979, 0.1845, 0.2083])
    ])
    valid_tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.2570, 0.2570, 0.2570], std=[0.3372, 0.3372, 0.3372]),
    ])
    
    train_data = Mnistm_dataset("hw2_data/digits/mnistm/data", train_file, train_label, train_tfm, domain=True)
    dataset = USPS_dataset("hw2_data/digits/usps/data", valid_file, valid_label, valid_tfm, domain=True)
    # print(dataset.__len__())
    train_loader = DataLoader(train_data, batch_size=128, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_model = FE().to(DEVICE)
    pred_model = LP().to(DEVICE)
    
    model_ckpt = torch.load("modelE.pt")
    feature_model.load_state_dict(model_ckpt["feature_model"])
    
    feature_model.eval()
    print(dataset.__len__())
    with torch.no_grad():
        features = []
        labels = []
        for data, label_d, label_y in dataloader:
            
            print(data.shape)
            data = data.to(DEVICE)
            label_d = label_d.to(DEVICE)
            
            feature = feature_model(data)
            features.extend(feature.cpu().numpy())
            labels.extend(label_d.cpu().numpy())
            
        for data, label_d, label_y in train_loader:
            
            print(data.shape)
            data = data.to(DEVICE)
            label_d = label_d.to(DEVICE)
            
            feature = feature_model(data)
            features.extend(feature.cpu().numpy())
            labels.extend(label_d.cpu().numpy())
            
        features = np.array(features)
        labels = np.array(labels)
        
        tsne = TSNE(n_components=2, init='random')
        features_tsne = tsne.fit_transform(features)
        
        plt.figure()
        for label in np.unique(labels):
            plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=label, s=3)
        
        plt.title("t-SNE Figure for USPS Domains")
        plt.legend()
        plt.grid(True)
        plt.savefig("plot/USPS-domain-tSNE")
        