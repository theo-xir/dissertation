import torch
import torch.nn as nn

def double_conv(in_channels, out_channels, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=padding),
       # nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=padding),
       # nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, channels:int):
        super().__init__()
                
        self.dconv_down1 = double_conv(channels, 64,1)
        self.dconv_down2 = double_conv(64, 128,1)
        self.dconv_down3 = double_conv(128, 256,1)
        self.dconv_down4 = double_conv(256, 512,1)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256,1)
        self.dconv_up2 = double_conv(128 + 256, 128,1)
        self.dconv_up1 = double_conv(128 + 64, 64,1)
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        # print(conv1.shape)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        # print(conv2.shape)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        # print(conv3.shape)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)    
        # print(x.shape)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)    
        # print(x.shape)    
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)   
        # print(x.shape)     
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
        
model = torch.load('state_dict_model.pt',map_location=torch.device('cpu'))

tsr=model.dconv_down1[0].weight
print(model)
print(tsr.shape)
averages=[sum(tsr[:,i,:,:]) for i in range(6)]
print(averages)