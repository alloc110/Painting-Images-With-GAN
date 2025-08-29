import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.conv1 = self._block(64, 128)
        self.conv2 = self._block(128, 256)
        self.conv3 = self._block(256, 512, stride=1)

        self.final_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def _block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.final_conv(x)