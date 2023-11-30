import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform

import torch
import torch.nn as nn
import torchvision.transforms as transform


class FeatureExtractor(nn.Module):
    def __init__(self, chin=3):
        super(FeatureExtractor, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(chin, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.ReLU(),
        )
        
    def forward(self, x):
    
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        
        return nn.Flatten()(out)
    

    
class LabelPredictor(nn.Module):
    def __init__(self, feature=6272):
        super(LabelPredictor, self).__init__()
        
        self.fc4 = nn.Sequential(
            nn.Linear(feature, 3072),
            nn.ReLU(),
        )
        
        self.fc5 = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.ReLU(),
        )
        
        self.fc6 = nn.Sequential(
            nn.Linear(2048, 10),
            # nn.Softmax(dim=1),                    # default dim = 1
        )
        
    def forward(self, x):
        x = self.fc4(x)
        x = self.fc5(x)
        out = self.fc6(x)
        
        return out
    

# https://blog.csdn.net/tsq292978891/article/details/79364140
# & chatgpt

class GradientReversalLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272, 1024),
            nn.ReLU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),               
        )
    
    def forward(self, x, Lambda):
        
        x = GradientReversalLayerFunction.apply(x, Lambda)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        
        return out
# FE = FeatureExtractor()
# LP = LabelPredictor()
# x = torch.randn((3, 3, 28, 28))   
# print(LP(FE(x)).shape)    #(3, 6272) = (3, 128, 7, 7)
# # print(7*7*128)