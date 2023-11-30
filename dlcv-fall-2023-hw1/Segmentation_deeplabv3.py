import torch
from torch.utils.data import Dataset, DataLoader
from p3_dataloader import SegmentationDataset
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
import numpy as np
from torch.utils.tensorboard import SummaryWriter
    
class Poly1Loss(nn.Module):
    def __init__(self):
        super(Poly1Loss, self).__init__()
        self.epsilon = 2
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=6)
        
    def forward(self, pred, target):
        
        ce_loss = self.cross_entropy(pred, target)                      # -logpt  so the formula is epsilon * (1 - pt)
        poly1loss = ce_loss + self.epsilon * (1 - torch.exp(-ce_loss))   # change to epsilon * (1 - torch(-ce_loss))
        return poly1loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.6, ignore_index=6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index)  #ignore the unknown

    def forward(self, logits, labels):
        log_pt = -self.CE(logits, labels)
        loss = -((1 - torch.exp(log_pt)) ** self.gamma) * self.alpha * log_pt
        return loss



def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = []
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        if (tp_fp + tp_fn - tp) == 0:    # causing the denominator to be zero can't divide by zero
            continue
        iou = tp / (tp_fp + tp_fn - tp)
        # in TA code can't dealt with the nan problem so we have to eliminate this problem
        # print(tp, tp_fn, tp_fp)
        mean_iou.append(iou)
    #     print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)

    return sum(mean_iou) / len(mean_iou)


# at first want to add this fc layer to the deeplab but it seem not work
# class DeepLab_fc(nn.Module):
#     def __init__(self):
#         super(DeepLab_fc, self).__init__()
#         self.deeplab = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
#         self.fc = nn.Linear(21, 7)
        
#     def forward(self, x):
#         x = self.deeplab(x)['out']
#         x = torch.permute(x, dims=(0, 2, 3, 1))
#         out = self.fc(x)
#         out = torch.permute(out, dims=(0, 3, 1, 2))
#         return out
    
    
# Hyper-paramter
NUM_EPOCH = 80
lr = 0.01
BATCH_SIZE = 4
best_mIoU = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train = SegmentationDataset("hw1_data/p3_data/train")
valid = SegmentationDataset("hw1_data/p3_data/validation")
# print(train.__len__(), valid.__len__()) # correct got 2000 train data and 257 valid data

train_laoder = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid, batch_size=1, shuffle=False, num_workers=4)

# https://learnopencv.com/deeplabv3-ultimate-guide/?fbclid=IwAR1adXRwsptR1XADF80gigi7TtMJ3ZrtdMi6eQOTjuM9QgS478YxPMLWu1s#What-Is-DeepLabv3?
# https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
# the output is output['out'] is of shape (N, 21, H, W)
# so I add conv to get the channel to 7(num_cls)

model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, n_classes=7)
model.classifier[4] = nn.Conv2d(256, 7, 1, 1)
model.aux_classifier[4] = nn.Conv2d(256, 7, 1, 1)
loss_fn = Poly1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# https://zhuanlan.zhihu.com/p/387162205
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_laoder), epochs=NUM_EPOCH)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH)

writer = SummaryWriter()
        
if __name__ == "__main__":

    model.to(DEVICE)

    for ep in range(1, NUM_EPOCH+1):
        print(f"Epoch {ep}")
        model.train()
        train_loss = []
        train_pred = []
        train_gt = []
        for batch_idx, (data, mask) in tqdm(enumerate(train_laoder)):
            
            data = data.to(DEVICE)
            mask = mask.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()
            
            output = model(data)
            
            pred, aux = output['out'], output['aux']
            loss = loss_fn(pred, mask) + loss_fn(aux, mask)
            train_loss.append(loss.item())
            # print(loss.item(), end=" ")
            
            # if ep % 6 == 0:
            #     pred = torch.argmax(pred, dim=1)
            #     pred = pred.detach().cpu().numpy()
            #     mask = mask.detach().cpu().numpy()
            #     train_pred.extend(pred)
            #     train_gt.extend(mask)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # if ep % 6 == 0:
        #     train_pred = np.array(train_pred)
        #     train_gt = np.array(train_gt)
        #     print("mIoU", mean_iou_score(train_pred, train_gt))
        #     writer.add_scalar("mIoU/train", mean_iou_score(train_pred, train_gt), ep)
            

        print("Loss", sum(train_loss) / len(train_loss))
        writer.add_scalar("loss/train", sum(train_loss) / len(train_loss), ep)
        
        
        with torch.no_grad():
            
            model.eval()
            valid_pred = []
            valid_gt = []
            
            for (data, label) in valid_loader:
                
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = model(data)['out']
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
                torch.save(model.state_dict(), "seg_model/best_model.pt")
            
                
            
            
            

# if __name__ == "__main__":
#     model.load_state_dict(torch.load("seg_model/best.pt"))
#     model.to(DEVICE)
#     model.eval()
#     mIoU = []
#     for data, mask in valid_loader:
#         data = data.to(DEVICE)
#         mask = mask.to(DEVICE)
        
#         pred = model(data)['out']
#         pred = torch.argmax(pred, dim=1)
#         pred = pred.detach()
#         mask = mask.detach()
#         mIoU.append(mean_iou_score(pred.cpu().numpy(), mask.cpu().numpy()))
#     print(sum(mIoU) / len(mIoU))