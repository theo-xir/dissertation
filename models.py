import torch.nn as nn
import torch

def double_conv(in_channels, out_channels, padding, norm):
    if norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )  
    else:
         return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding),
            nn.ReLU(inplace=True)
        )  



class UNetSmall(nn.Module):

    def __init__(self, channels:int,norm=False,noskip=False, transconv=False):
        super().__init__()
                
        self.dconv_down1 = double_conv(channels, 64,1, norm)
        self.dconv_down2 = double_conv(64, 128,1,norm)
        self.dconv_down3 = double_conv(128, 256,1,norm)
        self.dconv_down4 = double_conv(256, 512,1,norm)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        if noskip:
            if transconv:
                self.transconv_up3 = nn.ConvTranspose2d(512,256,2,2)
                self.transconv_up2 = nn.ConvTranspose2d(128,64,2,2)
                self.transconv_up1 = nn.ConvTranspose2d(32,16,2,2)
                self.dconv_up3 = double_conv(256, 128,1,norm)
                self.dconv_up2 = double_conv(64,32,1,norm)
                self.dconv_up1 = double_conv(16,8,1,norm)
                self.conv_last = nn.Conv2d(8, 1, 1)
            else:
                self.dconv_up3 = double_conv(512, 256,1,norm)
                self.dconv_up2 = double_conv(256, 128,1,norm)
                self.dconv_up1 = double_conv(128 , 64,1,norm)
                self.conv_last = nn.Conv2d(64, 1, 1)
        else:
            if transconv:
                self.transconv_up3 = nn.ConvTranspose2d(512,256,2,2)
                self.transconv_up2 = nn.ConvTranspose2d(256,128,2,2)
                self.transconv_up1 = nn.ConvTranspose2d(128,64,2,2)
                self.dconv_up3 = double_conv(512, 256,1,norm)
                self.dconv_up2 = double_conv(256, 128,1,norm)
                self.dconv_up1 = double_conv(128 , 64,1,norm)
            else:
                self.dconv_up3 = double_conv(256 + 512, 256,1,norm)
                self.dconv_up2 = double_conv(128 + 256, 128,1,norm)
                self.dconv_up1 = double_conv(128 + 64, 64,1,norm)
            self.conv_last = nn.Conv2d(64, 1, 1)
        
        
        
        
    def forward(self, x,noskip=False, transconv=False):
        # print(x.shape)
        conv1 = self.dconv_down1(x)
        # print(conv1.shape)
        x = self.maxpool(conv1)
        # print(x.shape)
        conv2 = self.dconv_down2(x)
        # print(conv2.shape)
        x = self.maxpool(conv2)
        # print(x.shape)        
        conv3 = self.dconv_down3(x)
        # print(conv3.shape)
        x = self.maxpool(conv3)  
        # print(x.shape)        
        x = self.dconv_down4(x)
        # print(x.shape)     
        if not transconv:   
            x = self.upsample(x)    
            # print(x.shape)
        else:
            x=self.transconv_up3(x)
            # print(x.shape)
        if not noskip:
            x = torch.cat([x, conv3], dim=1)
            # print(x.shape)
        x = self.dconv_up3(x)
        # print(x.shape)
        if not transconv:
            x = self.upsample(x)    
            # print(x.shape)
        else:
            x=self.transconv_up2(x)
            # print(x.shape)
        if not noskip:
            x = torch.cat([x, conv2], dim=1)     
            # print(x.shape) 
        x = self.dconv_up2(x)
        # print(x.shape)
        if not transconv:
            x = self.upsample(x)    
            # print(x.shape)
        else:
            x=self.transconv_up1(x)
            # print(x.shape)
        if not noskip:  
            x = torch.cat([x, conv1], dim=1)   
            # print(x.shape)        
        x = self.dconv_up1(x)
        # print(x.shape)
        
        out = self.conv_last(x)
        # print(out.shape)
        return out
