import torch
from torch.utils.data import Dataset, DataLoader
from p3_dataloader import SegmentationDataset
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch.nn as nn
from tqdm import tqdm
from vgg16_fcn32s import VGG16_FCN32
from torch.utils.tensorboard import SummaryWriter
import numpy as np

    
def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = []
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        if tp == 0 and tp_fn == 0 and tp_fp == 0:
            continue
        iou = tp / (tp_fp + tp_fn - tp)
        # in TA code can't dealt with the nan problem so we have to eliminate this problem
        # print(tp, tp_fn, tp_fp)
        mean_iou.append(iou)
    #     print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)

    return sum(mean_iou) / len(mean_iou)

# Hyper-paramter
NUM_EPOCH = 50
lr = 0.001
BATCH_SIZE = 4
best_mIoU = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_fn = nn.CrossEntropyLoss()
model = VGG16_FCN32()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

train = SegmentationDataset("hw1_data/p3_data/train")
valid = SegmentationDataset("hw1_data/p3_data/validation")

train_laoder = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

writer = SummaryWriter()

if __name__ == "__main__":
    # print(model)
    model.to(DEVICE)
    print(train.__len__(), valid.__len__())
    
    for ep in range(1, NUM_EPOCH+1):
        print(f"Epoch {ep}")
        model.train()
        train_loss = []
        train_mIoU = []
        
        for batch_idx, (data, mask) in tqdm(enumerate(train_laoder)):
            
            data = data.to(DEVICE)
            mask = mask.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()
            
            pred = model(data)
            
            loss = loss_fn(pred, mask)
            train_loss.append(loss.item())
            
            if ep % 10 == 0:
                pred = torch.argmax(pred, dim=1)
                pred = pred.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()
                train_mIoU.append(mean_iou_score(pred, mask))
            # print(loss.item(), end=" ")
            loss.backward()
            optimizer.step()
            

        print("Loss", sum(train_loss) / len(train_loss))
        writer.add_scalar("loss/train", sum(train_loss) / len(train_loss), ep)
        if ep % 10 == 0:
            print("Train mIoU", sum(train_mIoU) / len(train_mIoU))
            writer.add_scalar("mIoU/train", sum(train_mIoU) / len(train_mIoU), ep)
            
            
        with torch.no_grad():
            
            model.eval()
            valid_pred = []
            valid_gt = []
            
            for (data, label) in valid_loader:
                
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = model(data)
                pred = torch.argmax(pred, dim=1)
                
                pred = pred.cpu().numpy()
                label = label.cpu().numpy()
                
                valid_pred.extend(pred)
                valid_gt.extend(label)
                
                
            valid_pred = np.array(valid_pred)
            valid_gt = np.array(valid_gt)
            mIoU = mean_iou_score(valid_pred, valid_gt)
            writer.add_scalar("mIoU/valid", mIoU, ep)
            print("Valid", mIoU)
            if mIoU >= best_mIoU:
                print("Save Model")
                best_mIoU = mIoU
                torch.save(model.state_dict(), "fcn32.pt")
            