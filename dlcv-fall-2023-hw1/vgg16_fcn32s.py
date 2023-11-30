import torch.nn as nn
import torchvision

class VGG16_FCN32(nn.Module):
    def __init__(self):
        super(VGG16_FCN32, self).__init__()
        
        self.backbone = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.fcn32 = nn.ConvTranspose2d(512, 7, kernel_size=32, stride=32)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return self.fcn32(x)
    
# model = VGG16_FCN32().to('cuda')
# print(model(torch.randn(32, 3, 512, 512).to('cuda')).shape)
# # got same shape (32, 3, 512, 512)
# model = timm.create_model("vgg11", pretrained=True).features.to("cuda")
# out = model(torch.randn(32, 3, 512, 512).to('cuda'))
# out = nn.ConvTranspose2d(512, 7, kernel_size=32, stride=32).to('cuda')(out)
# print(out.shape)
# model = torchvision.models.vgg16()
# print(model)