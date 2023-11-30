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

    train_tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop((32, 32), padding=4),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    valid_tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Hyper Parameter
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    BATCH_SIZE = 128
    lr = 0.001
    NUM_EPOCH = 500

    train = p1_dataset(root="hw1_data/p1_data/train_50", transform=train_tfm)
    valid = p1_dataset(root="hw1_data/p1_data/val_50", transform=valid_tfm)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = VGG13().to(DEVICE)
    # model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.DEFAULT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    plot = [1, 250, 500]
    
    for ep in range(1, NUM_EPOCH+1):
        
        print(f"Epoch {ep}")
        train_loss = list()
        train_acc = list()
        model.train()
        
        for batch_idx, (data, label) in enumerate(train_loader):
            
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            
            pred = torch.argmax(pred, dim=1)
            # print(pred.shape)
            train_loss.append(loss.item())
            train_acc.append(torch.mean((pred == label).float()).item())
        
        
        mean_train_loss = sum(train_loss) / len(train_loss)
        mean_train_acc = sum(train_acc) / len(train_acc)
        print("Loss", mean_train_loss)
        writer.add_scalar("Loss/Train", mean_train_loss, ep)
        writer.add_scalar("Acc/Train", mean_train_acc, ep)
        
        model.eval()
        valid_loss = list()
        valid_acc  = list()
        
        with torch.no_grad():
            for data, label in valid_loader:
                
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                pred = model(data)
                valid_loss.append(criterion(pred, label).item())
                pred = torch.argmax(pred, dim=1)
                valid_acc.append(torch.mean((pred == label).float()).item())
            
            mean_valid_loss = sum(valid_loss) / len(valid_loss)
            mean_valid_acc = sum(valid_acc) / len(valid_acc)
            writer.add_scalar("Loss/Valid", mean_valid_loss, ep)
            writer.add_scalar("Acc/Valid", mean_valid_acc, ep)
            
            if best_acc <= mean_valid_acc:
                print("Save Model")
                best_acc = mean_valid_acc
                torch.save(model.state_dict(), "cls/best_model.ckpt")
                print(best_acc)

            if ep in plot:
                
                # plot the PCA and t-SNE
                features = list()
                labels = list()
                for data, label in valid_loader:
                    
                    data = data.to(DEVICE)
                    feature = model.get_embedding(data)
                    features.extend(feature.cpu().numpy())
                    labels.extend(label.cpu().numpy())
                
                features = np.array(features)
                labels = np.array(labels)
                pca = PCA(n_components=2)
                features_pca = pca.fit_transform(features)
                
                plt.figure()
                for label in np.unique(labels):
                    # index is labels == label due to we want to see the gt cls feature is close
                    plt.scatter(features_pca[labels == label, 0], features_pca[labels == label, 1], label=label, s=5)    
                plt.title(f"PCA figure for epoch {ep}")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"./p1plots/PCA_epoch{ep}")

                tsne = TSNE(n_components=2, init='random')
                features_tsne = tsne.fit_transform(features)
                
                plt.figure()
                for label in np.unique(labels):
                    plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=label, s=5)
                plt.title(f"t-SNE figure for epoch {ep}")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"./p1plots/t-SNE_epoch{ep}")
                
                
                
                
                
            
            
            
            
    
    
# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#     return

# if __name__ == "__main__":

#     train = p1_dataset(root="hw1_data/p1_data/train_50", transform=train_tfm)
#     train_loader = DataLoader(train, batch_size=128, shuffle=False, num_workers=4)
#     train_loader = iter(train_loader)
#     images, labels = next(train_loader)
#     print(images.shape, labels.shape)
    
#     # show images
#     imshow(torchvision.utils.make_grid(images))


