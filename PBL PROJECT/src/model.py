import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class MultispectralUNet(nn.Module):
    """
    Multispectral U-Net for Satellite Semantic Segmentation
    Matches exactly the architecture trained in Google Colab.
    Input: [Batch, 4, H, W] (Blue, Green, Red, NIR)
    Output: [Batch, 6, H, W] (Drought Severity Classes 0 to 5)
    """
    def __init__(self, in_channels=4, num_classes=6):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(128 + 64, 64)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = DoubleConv(64 + 32, 32)
        
        self.outc = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        return self.outc(x)

def get_model(in_channels=4, num_classes=6, weights_path=None, device='cpu'):
    """Factory function to build and optionally load model weights."""
    model = MultispectralUNet(in_channels=in_channels, num_classes=num_classes)
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
