import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision
import pandas as pd
import matplotlib.pyplot as plt

class Mnistm_dataset(Dataset):
    def __init__(self, root, filename, label, transform=None):
        
        super(Mnistm_dataset, self).__init__()
        index = np.argsort(filename)
        self.root = root
        self.labels = label[index]
        filename = filename[index]
        # print(len(self.labels), len(filename))             # 44800
        self.filenames = [os.path.join(root, fn) for fn in filename]
        self.transform = transform
        self.len = len(self.filenames)
        
    def __getitem__(self, index):
        image_fn = self.filenames[index]
        label = self.labels[index]
        img = Image.open(image_fn)
        
        if self.transform:
            img = self.transform(img)
        

        return img, label
    
    def __len__(self):
        return self.len
        

def inverse_transform(img):
    return ((img.clamp(-1, 1) + 1.0) / 2.0) * 255.0
    
# tfm = transforms.Compose([
#         transforms.ToTensor(),      # range [0, 1]
# #         transforms.Lambda(lambd=lambda t: (t.clamp(-1, 1) * 2) - 1),     # scale to [-1, 1]
#     ])    



# train_pd = pd.read_csv("hw2_data/digits/mnistm/train.csv")
# train_filename = train_pd["image_name"].to_numpy()
# train_label = train_pd["label"].to_numpy()

# train_data = Mnistm_dataset("hw2_data/digits/mnistm/data", train_filename, train_label, tfm)
# train_dataloader = DataLoader(train_data, batch_size=16, shuffle=False, num_workers=1)

# if __name__ == "__main__":
    
#     for data, domain, label in train_dataloader:
#         print(data, domain, label)
#         break
        # print(data.shape)
        # print(label.shape)
    #         print(data.shape)
    #         print(torchvision.utils.make_grid(data).shape)
    #         plt.imshow(np.array(inverse_transform(torchvision.utils.make_grid(data).permute(dims=(1, 2, 0))))/255)
    #         plt.show()
    #     print(train_data.__len__())
    # mean = torch.zeros(3)
    # std = torch.zeros(3)

    # for i, _ in train_data:
    #     mean += i.mean(dim=(1,2))
    #     std += i.std(dim=(1,2))
    # mean /= len(train_data)
    # std /= len(train_data)

    # print(mean, std)
    # tensor([0.4631, 0.4666, 0.4195]) tensor([0.1979, 0.1845, 0.2083])