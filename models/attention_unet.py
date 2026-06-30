"""
Attention U-Net: U-Net với Attention Gates ở skip connections.
Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def x2conv(in_channels, out_channels, inner_channels=None):
    """Khối tích chập đôi (Conv-BN-ReLU) x2."""
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    """
    Attention Gate: lọc thông tin skip connection bằng tín hiệu gating từ decoder.
    - x: skip feature từ encoder (batch, C_x, H, W)
    - g: gating feature từ decoder (batch, C_g, h, w)
    """
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2

        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True)
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        x1 = self.W_x(x)
        g1 = self.W_g(g)

        if x1.shape[2:] != g1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        f = self.relu(x1 + g1)
        psi = self.psi(f)
        return x * psi


class Encoder(nn.Module):
    """Encoder block: x2conv + MaxPool."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled  # skip feature, pooled output


class Decoder(nn.Module):
    """Decoder block: Upsample + Attention Gate + Concat + x2conv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.att_gate = AttentionGate(
            in_channels=in_channels,
            gating_channels=in_channels // 2,
        )
        self.up_conv = x2conv(in_channels + in_channels // 2, out_channels)

    def forward(self, skip_feat, x):
        x = self.up(x)
        if x.shape[2:] != skip_feat.shape[2:]:
            x = F.interpolate(x, size=skip_feat.shape[2:], mode='bilinear', align_corners=True)

        skip_att = self.att_gate(skip_feat, x)
        x = torch.cat([skip_att, x], dim=1)
        x = self.up_conv(x)
        return x


class AttentionUNet(nn.Module):
    """
    Attention U-Net.
    Args:
        num_classes: số lớp đầu ra (1 cho binary segmentation).
        in_channels: số kênh đầu vào (3=RGB, 4=RGB+NDVI).
        freeze_bn: đóng băng BatchNorm hay không.
    """
    def __init__(self, num_classes=1, in_channels=4, freeze_bn=False):
        super().__init__()

        # Encoder
        self.start_conv = x2conv(in_channels, 64)
        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024)

        # Bottleneck
        self.bottleneck = x2conv(1024, 1024)

        # Decoder
        self.dec1 = Decoder(1024, 512)
        self.dec2 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec4 = Decoder(128, 64)

        # Final
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self._initialize_weights()
        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.start_conv(x)
        skip1, x2p = self.enc1(x1)
        skip2, x3p = self.enc2(x2p)
        skip3, x4p = self.enc3(x3p)
        skip4, x5p = self.enc4(x4p)

        x_b = self.bottleneck(x5p)

        x = self.dec1(skip4, x_b)
        x = self.dec2(skip3, x)
        x = self.dec3(skip2, x)
        x = self.dec4(skip1, x)

        return torch.sigmoid(self.final_conv(x))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
