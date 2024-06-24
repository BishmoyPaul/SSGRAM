from S2FN_3D.SE_3D import SE_3D
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSPSA(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(SSPSA, self).__init__()
        self.sc = 3

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.c1 = nn.Conv3d(output_channels, output_channels//self.sc, (1,1,7), stride=1, padding=(0,0,7//2), dilation=1,  bias=False)
        self.c2 = nn.Conv3d(output_channels, output_channels//self.sc, (3,3,1), stride=1, padding=(3//2,3//2,0), dilation=1,  bias=False)
        self.c3 = nn.Conv3d(output_channels, output_channels//self.sc, (3,3,7), stride=1, padding=(3//2,3//2,7//2), dilation=1,  bias=False)

        self.bottle = nn.Conv3d(input_channels, output_channels, (3,3,7), stride=(1,1,2), padding=(3//2,3//2,0), dilation=1,  bias=False)
        self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.1, affine=True)

        self.se = SE_3D(output_channels//self.sc)

    def forward(self, x):
        x1 = self.bottle(x)
        x1 = F.relu(self.bn(x1))

        bs, c, h, w, b = x1.size()


        out1 = self.c1(x1)
        out2 = self.c2(x1)
        out3 = self.c3(x1)

        feats = torch.cat((out1, out2, out3), dim=1)
        feats = feats.view(bs, self.sc, self.output_channels//self.sc, h, w, b)

        se1 = self.se(out1)
        se2 = self.se(out2)
        se3 = self.se(out3)


        x_epsa = torch.cat((se1, se2, se3), dim=1)
        attention_vectors = x_epsa.view(bs, self.sc, self.output_channels//self.sc, 1, 1)
        attention_vectors = F.softmax(attention_vectors, dim = 1)

        feats_weight = feats * attention_vectors.unsqueeze(-1)

        out = feats_weight.view(bs, self.output_channels, h, w, b)

        out = out + x1

        return out