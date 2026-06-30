import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attn.view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_pool, max_pool], dim=1)
        attn = self.sigmoid(self.conv(cat))
        return x * attn

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x

class ChannelAttentionPriority(nn.Module):
    """Channel Attention with Priority for specific channels (e.g., NDVI)"""
    def __init__(self, in_channels, reduction=16, ndvi_idx=3, ndvi_factor=2.0):
        super().__init__()
        self.ndvi_idx = ndvi_idx         # index của kênh NDVI, 0-based
        self.ndvi_factor = ndvi_factor   # hệ số ưu tiên
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 1) Global Pooling Avg & Max
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)

        # 2) Ưu tiên NDVI
        avg_pool_clone = avg_pool.clone()
        max_pool_clone = max_pool.clone()
        avg_pool_clone[:, self.ndvi_idx] = avg_pool_clone[:, self.ndvi_idx] * self.ndvi_factor
        max_pool_clone[:, self.ndvi_idx] = max_pool_clone[:, self.ndvi_idx] * self.ndvi_factor

        # 3) Qua MLP & Sigmoid
        attn = self.sigmoid(self.mlp(avg_pool_clone) + self.mlp(max_pool_clone))
        return x * attn.view(b, c, 1, 1)

class CBAMPriority(nn.Module):
    """CBAM with Channel Attention Priority"""
    def __init__(self, in_channels, reduction=16, kernel_size=7, ndvi_idx=3, ndvi_factor=2.0):
        super().__init__()
        self.channel_attn = ChannelAttentionPriority(in_channels, reduction, ndvi_idx, ndvi_factor)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x

def x2conv(in_channels, out_channels, inner_channels=None, use_cbam=False, priority=False):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    layers = [
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if use_cbam:
        if priority:
            layers.append(CBAMPriority(out_channels))
        else:
            layers.append(CBAM(out_channels))
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_cbam=False, priority=False):
        super().__init__()
        self.conv = x2conv(in_channels, out_channels, use_cbam=use_cbam, priority=priority)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

    def forward(self, x):
        skip = self.conv(x)
        x_pooled = self.pool(skip)
        return skip, x_pooled

class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_cbam=False, priority=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        total_channels = (in_channels // 2) + skip_channels
        self.up_conv = x2conv(total_channels, out_channels, use_cbam=use_cbam, priority=priority)

    def forward(self, skip, x):
        x = self.up(x)
        if x.size()[-2:] != skip.size()[-2:]:
            x = F.interpolate(x, size=skip.size()[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        x = self.up_conv(x)
        return x

class UNetCBAM(nn.Module):
    """
    U-Net with Convolutional Block Attention Module (CBAM)
    """
    def __init__(self, num_classes=1, in_channels=4, freeze_bn=False, use_cbam=True):
        super().__init__()
        self.start_conv = x2conv(in_channels, 64, use_cbam=use_cbam)
        self.enc1 = Encoder(64, 128, use_cbam=use_cbam)
        self.enc2 = Encoder(128, 256, use_cbam=use_cbam)
        self.enc3 = Encoder(256, 512, use_cbam=use_cbam)
        self.enc4 = Encoder(512, 1024, use_cbam=use_cbam)

        self.bottleneck = x2conv(1024, 1024, use_cbam=use_cbam)

        self.dec1 = Decoder(1024, skip_channels=1024, out_channels=512, use_cbam=use_cbam)
        self.dec2 = Decoder(512, skip_channels=512, out_channels=256, use_cbam=use_cbam)
        self.dec3 = Decoder(256, skip_channels=256, out_channels=128, use_cbam=use_cbam)
        self.dec4 = Decoder(128, skip_channels=128, out_channels=64, use_cbam=use_cbam)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()
        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        x0 = self.start_conv(x)
        s1, x1 = self.enc1(x0)
        s2, x2 = self.enc2(x1)
        s3, x3 = self.enc3(x2)
        s4, x4 = self.enc4(x3)

        x_mid = self.bottleneck(x4)

        x = self.dec1(s4, x_mid)
        x = self.dec2(s3, x)
        x = self.dec3(s2, x)
        x = self.dec4(s1, x)

        return torch.sigmoid(self.final_conv(x))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class UNetCBAMPriority(UNetCBAM):
    """
    U-Net with CBAM and Priority on specific channel (e.g. NDVI).
    """
    def __init__(self, num_classes=1, in_channels=4, freeze_bn=False, use_cbam=True):
        nn.Module.__init__(self)
        self.start_conv = x2conv(in_channels, 64, use_cbam=use_cbam, priority=True)
        self.enc1 = Encoder(64, 128, use_cbam=use_cbam, priority=True)
        self.enc2 = Encoder(128, 256, use_cbam=use_cbam, priority=True)
        self.enc3 = Encoder(256, 512, use_cbam=use_cbam, priority=True)
        self.enc4 = Encoder(512, 1024, use_cbam=use_cbam, priority=True)

        self.bottleneck = x2conv(1024, 1024, use_cbam=use_cbam, priority=True)

        self.dec1 = Decoder(1024, skip_channels=1024, out_channels=512, use_cbam=use_cbam, priority=True)
        self.dec2 = Decoder(512, skip_channels=512, out_channels=256, use_cbam=use_cbam, priority=True)
        self.dec3 = Decoder(256, skip_channels=256, out_channels=128, use_cbam=use_cbam, priority=True)
        self.dec4 = Decoder(128, skip_channels=128, out_channels=64, use_cbam=use_cbam, priority=True)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()
        if freeze_bn:
            self.freeze_bn()
