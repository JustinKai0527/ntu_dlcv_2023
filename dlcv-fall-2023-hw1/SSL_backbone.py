import torch
from byol_pytorch import BYOL
from torchvision import models
from SSL_dataloader import MiniDataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
 
if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCH = 1000

    resnet = models.resnet50().to(DEVICE)

    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool'
    ).to(DEVICE)

    tfm = transforms.Compose([
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
            mean=torch.tensor([0.4705, 0.4495, 0.4037]),
            std=torch.tensor([0.2170, 0.2149, 0.2145])),
    ])


    dataset = MiniDataset("hw1_data/p2_data/mini/train", tfm)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH * len(data_loader))

    for ep in range(30):
        print(ep)
        
        total_loss = []
        for data in tqdm(data_loader):
            images = data.to(DEVICE)
            loss = learner(images)
            optimizer.zero_grad()
            loss.backward()
            total_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
            learner.update_moving_average() # update moving average of target encoder
        print(sum(total_loss) / len(total_loss))
        
    # save your improved network
    torch.save(resnet.state_dict(), './backbone_resnet.pt')