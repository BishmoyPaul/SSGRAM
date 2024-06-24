from S2FN_3D.EPSA import EPSA
import torch.nn as nn
import torch.nn.functional as F

class Residual_EPSA(nn.Module):
    def __init__(
            self,
            in_channels,out_channels,kernel_size,padding,
            use_1x1conv=False,stride=1,start_block=False,end_block=False,
    ):
        super(Residual_EPSA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride), nn.ReLU())
        self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        if not start_block:
            self.bn0 = nn.BatchNorm3d(in_channels)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if start_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        if end_block:
            self.bn2 = nn.BatchNorm3d(out_channels)


        self.EPSA_mod = EPSA(out_channels)


        self.start_block = start_block
        self.end_block = end_block

    def forward(self, X):
        identity = X

        if self.start_block:
            out = self.conv1(X)
        else:
            out = self.bn0(X)
            out = F.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = F.relu(out)


        out = self.EPSA_mod(out)

        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out)



        out += identity

        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out)

        return out


