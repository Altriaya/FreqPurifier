import torch
import torch.nn as nn

class UNet1D(nn.Module):
    """
    Simple 1D U-Net for Signal Reconstruction / Mask Generation.
    Input: Amplitude Spectrum (B, 1, Len)
    Output: Purified Amplitude Spectrum (B, 1, Len)
    """
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(UNet1D, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 4, features * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = self._block(features * 8, features * 4)
        
        self.up2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(features * 4, features * 2)
        
        self.up1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = self._block(features * 2, features)
        
        self.final_conv = nn.Conv1d(features, out_channels, kernel_size=1)
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        b = self.bottleneck(p3)
        
        u3 = self.up3(b)
        if u3.shape[-1] != e3.shape[-1]:
             u3 = torch.nn.functional.interpolate(u3, size=e3.shape[-1])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self.up2(d3)
        if u2.shape[-1] != e2.shape[-1]:
             u2 = torch.nn.functional.interpolate(u2, size=e2.shape[-1])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape[-1] != e1.shape[-1]:
             u1 = torch.nn.functional.interpolate(u1, size=e1.shape[-1])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        return self.final_conv(d1)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=32):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(features, features * 2, 4, 2, 1),
            nn.BatchNorm1d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(features * 2, features * 4, 4, 2, 1),
            nn.BatchNorm1d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(features * 4, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
