import torch
import torch.nn as nn
import torch.nn.functional as F

def x2conv(in_channels, out_channels, inner_channels=None):
    """Khối tích chập kép (Conv-BN-ReLU) x2."""
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Encoder(nn.Module):
    """Encoder block: x2conv + MaxPool."""
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x

class Decoder(nn.Module):
    """Decoder block: Upsample + Concat + x2conv."""
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Khối tích chập xử lý kết quả sau khi nối với các đặc trưng từ encoder.
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, skip, x, interpolate=True):
        x = self.up(x)
        if (x.size(2) != skip.size(2)) or (x.size(3) != skip.size(3)):
            if interpolate:
                x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode="bilinear", align_corners=True)
            else:
                diffY = skip.size()[2] - x.size()[2]
                diffX = skip.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        x = torch.cat([skip, x], dim=1)
        x = self.up_conv(x)
        return x

class UNet(nn.Module):
    """
    Standard U-Net.
    Args:
        num_classes: số lớp đầu ra (1 cho binary segmentation).
        in_channels: số kênh đầu vào (3=RGB, 4=RGB+NDVI, 5=RGB+NDVI+Edge).
        freeze_bn: đóng băng BatchNorm hay không.
    """
    def __init__(self, num_classes=1, in_channels=4, freeze_bn=False):
        super(UNet, self).__init__()

        self.start_conv = x2conv(in_channels, 64)
        self.down1 = Encoder(64, 128)
        self.down2 = Encoder(128, 256)
        self.down3 = Encoder(256, 512)
        self.down4 = Encoder(512, 1024)

        self.middle_conv = x2conv(1024, 1024)

        self.up1 = Decoder(1024, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)
        self.up4 = Decoder(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self._initialize_weights()
        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x_mid = self.middle_conv(x5)

        x = self.up1(x4, x_mid)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.final_conv(x)
        return torch.sigmoid(x)

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
