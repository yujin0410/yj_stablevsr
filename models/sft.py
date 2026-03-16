# models/sft.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SFT_Module(nn.Module):
    def __init__(self, cond_channels=36, target_channels=320):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 128, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.conv_gamma = nn.Conv2d(128, target_channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(128, target_channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, feature, condition):
        shared = self.shared_conv(condition)
        gamma = torch.tanh(self.conv_gamma(shared)) * 0.1
        beta  = torch.tanh(self.conv_beta(shared))  * 0.1

        if feature.shape[2:] != gamma.shape[2:]:
            gamma = F.interpolate(gamma, size=feature.shape[2:], mode='bilinear', align_corners=False)
            beta  = F.interpolate(beta,  size=feature.shape[2:], mode='bilinear', align_corners=False)

        return feature * (1 + gamma) + beta
