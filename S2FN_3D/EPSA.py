from S2FN_3D.SE_3D import SE_3D
import torch
import torch.nn as nn
import torch.nn.functional as F


class EPSA(nn.Module):
    def __init__(self, input_channels):
        super(EPSA, self).__init__()
        self.sc = 4

        self.input_channels = input_channels

        self.c1 = nn.Conv3d(input_channels, input_channels//self.sc, (1,1,3), stride=1, padding=(0,0,3//2), dilation=1,  bias=False)
        self.c2 = nn.Conv3d(input_channels, input_channels//self.sc, (1,1,5), stride=1, padding=(0,0,5//2), dilation=1,  bias=False)
        self.c3 = nn.Conv3d(input_channels, input_channels//self.sc, (1,1,7), stride=1, padding=(0,0,7//2), dilation=1,  bias=False)
        self.c4 = nn.Conv3d(input_channels, input_channels//self.sc, (1,1,9), stride=1, padding=(0,0,9//2), dilation=1,  bias=False)

        self.se = SE_3D(input_channels//self.sc)

    def forward(self, x):

        bs, c, h, w, b = x.size()

        out1 = self.c1(x)
        out2 = self.c2(x)
        out3 = self.c3(x)
        out4 = self.c4(x)

        feats = torch.cat((out1, out2, out3, out4), dim=1)
        feats = feats.view(bs, 4, self.input_channels//self.sc, h, w, b)

        se1 = self.se(out1)
        se2 = self.se(out2)
        se3 = self.se(out3)
        se4 = self.se(out4)

        x_se = torch.cat((se1, se2, se3, se4), dim=1)
        attention_vectors = x_se.view(bs, 4, self.input_channels//self.sc, 1, 1)
        attention_vectors = F.softmax(attention_vectors, dim = 1)

        feats_weight = feats * attention_vectors.unsqueeze(-1)

        out = feats_weight.view(bs, c, h, w, b)


        return out