import torch
from torch.utils.data import Dataset, DataLoader
from p3_dataloader import SegmentationDataset, get_mask, imshow
from vgg16_fcn32s import VGG16_FCN32
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
import numpy as np
from torch.utils.tensorboard import SummaryWriter


myseed =  666 # set a random seed for reproducibility
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    
# class Poly1Loss(nn.Module):
#     def __init__(self):
#         super(Poly1Loss, self).__init__()
#         self.epsilon = 2
#         self.cross_entropy = nn.CrossEntropyLoss(ignore_index=6)
        
#     def forward(self, pred, target):
        
#         ce_loss = self.cross_entropy(pred, target)                      # -logpt  so the formula is epsilon * (1 - pt)
#         poly1loss = ce_loss + self.epsilon * (1 - torch.exp(-ce_loss))   # change to epsilon * (1 - torch(-ce_loss))
#         return poly1loss
    
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
NUM_EPOCH = 100
lr = 0.002
BATCH_SIZE = 4
best_mIoU = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


valid = SegmentationDataset("hw1_data/p3_data/validation")

valid_loader = DataLoader(valid, batch_size=1, shuffle=False, num_workers=4)

# https://learnopencv.com/deeplabv3-ultimate-guide/?fbclid=IwAR1adXRwsptR1XADF80gigi7TtMJ3ZrtdMi6eQOTjuM9QgS478YxPMLWu1s#What-Is-DeepLabv3?
# https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
# the output is output['out'] is of shape (N, 21, H, W)
# so I add conv to get the channel to 7(num_cls)

if __name__ == "__main__":
    
    # model = VGG16_FCN32()
    # model.load_state_dict(torch.load("fcn32.pt"))
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 7, 1, 1)
    model.aux_classifier[4] = nn.Conv2d(256, 7, 1, 1)
    model.load_state_dict(torch.load("model/seg_model/0.7467.pth"))
    print(model)
    model.to(DEVICE)  
    model.eval() 
    mIoU = []
    
    model.eval()
    with torch.no_grad():
        va_loss = 0
        all_preds = []
        all_gt = []
        for x, y in tqdm(valid_loader):
            x, y = x.to(DEVICE), y.to(DEVICE, dtype=torch.long)
            out = model(x)['out']
            pred = out.argmax(dim=1)

            # get_mask(pred.detach())
            # get_mask(y.detach())
            pred = pred.detach().cpu().numpy().astype(np.int64)
            y = y.detach().cpu().numpy().astype(np.int64)
            all_preds.append(pred)
            all_gt.append(y)


        mIoU = mean_iou_score(np.concatenate(all_preds, axis=0), np.concatenate(all_gt, axis=0))
        print(mIoU)
