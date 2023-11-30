import torch.nn as nn
import torch

class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        
        self.back_bone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # block 1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # block 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # block 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # block 4
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # block 5
        )
        
        # max-pool 5 times so the dim 512 x 1 x 1 flatten to 512
        self.flatten = nn.Flatten()
        self.fc6 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 512)
        )
        
        # we let the ReLU to be add to the fc8 due to hw1 need last 2 layer easier to get the output
        self.fc8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(512, 50)
        )
        
    def forward(self, x):
        x = self.flatten(self.back_bone(x))
        x = self.fc6(x)
        return self.fc8(x)
    
        # TA wants to get the last conv layer output
    def get_embedding(self, x):
        x = self.flatten(self.back_bone(x))
        return x
    

        
# if __name__ == "__main__":
#     test = torch.randn((100, 3, 32, 32)).to('cuda')
#     model = VGG13().to('cuda')
#     print(model(test).shape)