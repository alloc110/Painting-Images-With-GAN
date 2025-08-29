import torch
from torch import nn

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.initial_down = DownSampleBlock(in_channels, 64, batch_norm=False)
        self.down1 = DownSampleBlock(64, 128)
        self.down2 = DownSampleBlock(128, 256)
        self.down3 = DownSampleBlock(256, 512)
        self.down4 = DownSampleBlock(512, 512)
        self.down5 = DownSampleBlock(512, 512)
        self.down6 = DownSampleBlock(512, 512)

        self.bottleneck = DownSampleBlock(512, 512, batch_norm=False)

        self.up0 = UpSampleBlock(512, 512, dropout=True)
        self.up1 = UpSampleBlock(512 * 2, 512, dropout=True)
        self.up2 = UpSampleBlock(512 * 2, 512, dropout=True)
        self.up3 = UpSampleBlock(512 * 2, 512)
        self.up4 = UpSampleBlock(512 * 2, 256)
        self.up5 = UpSampleBlock(256 * 2, 128)
        self.up6 = UpSampleBlock(128 * 2, 64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)

        u0 = self.up0(bottleneck)
        u1 = self.up1(torch.cat([u0, d7], 1))
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))

        return self.final_up(torch.cat([u6, d1], 1))