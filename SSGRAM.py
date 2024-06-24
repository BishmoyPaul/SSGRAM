from S2FN_3D.Residual_EPSA import Residual_EPSA
from S2FN_3D.SSPSA import SSPSA
from GAFP.GAFP import GraphAttentionFeatureProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SSGRAM(nn.Module):
    def __init__(self, band, classes, reduction, channels, central_size=3, graph_correlation = 'cosine', attn_map = False,
                 feature_norm_method = 'normalize', mid_size = 128, graph_linear = True, depth = 3):
        super(SSGRAM, self).__init__()
        self.name = 'SSGRAM'

        self.depth = depth

        self.res_net1 = Residual_EPSA(channels, channels, (1, 1, 7), (0, 0, 3), start_block=True)
        self.res_net2 = Residual_EPSA(channels, channels, (3, 3, 1), (1, 1, 0))
        if self.depth >= 3:
            self.res_net3 = Residual_EPSA(channels, channels,(3, 3, 7), (1, 1, 3))
        if self.depth >= 4:
            self.res_net4 = Residual_EPSA(channels, channels,(3, 3, 7), (1, 1, 3))


        kernel_3d = math.ceil((band - 6) / 2)

        self.mid_size = mid_size
        self.conv2 = nn.Conv3d(in_channels=channels, out_channels=self.mid_size, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d),stride=(1, 1, 1))

        self.batch_norm2 = nn.Sequential(nn.BatchNorm3d(self.mid_size, eps=0.001, momentum=0.1, affine=True),  nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(in_channels=1,out_channels=channels,padding=(0, 0, 0),kernel_size=(3, 3, mid_size),stride=(1, 1, 1))

        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(channels, eps=0.001, momentum=0.1,affine=True),  nn.ReLU(inplace=True))

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))

        self.epsa_init = SSPSA(1, channels)


        feature = channels

        self.attn_map = attn_map
        self.attn_map_weight = nn.Parameter(torch.zeros(1))
        self.graph = GraphAttentionFeatureProcessor(band, feature=feature,central_size=central_size, correlation = graph_correlation, get_attn_map = attn_map)

        linear_inp = channels
        linear_inp = linear_inp + feature

        self.full_connection = nn.Sequential(nn.Linear(linear_inp, classes))

        self.feature_norm_method = feature_norm_method
        if self.feature_norm_method == 'bn':
            self.bn_g = nn.BatchNorm1d(feature)
            self.bn = nn.BatchNorm1d(channels)


        print(f"Using only {self.depth} EPSA layers")

    def forward(self, X):
        x1 = self.epsa_init(X)

        x2 = self.res_net1(x1)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))

        map = None
        if self.attn_map:
          x1_graph, map = self.graph(X.squeeze(1))
          x2 = x2*(1-self.attn_map_weight) + self.attn_map_weight* x2 * map.unsqueeze(-1)
        else:
          x1_graph = self.graph(X.squeeze(1))


        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        if self.depth == 2:
            x3 = x2
        else:
            x3 = self.res_net3(x2)
            if self.depth >= 4:
                x3 = self.res_net4(x3)

        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)


        if self.feature_norm_method == 'norm':
            x4 = F.normalize(x4, dim = -1)
            x1_graph = F.normalize(x1_graph, dim = -1)

        elif self.feature_norm_method == 'bn':
            x4 = self.bn(x4)
            x1_graph = self.bn_g(x1_graph)

        else:
            pass

        x4 = torch.cat([x4, x1_graph], dim=1)

        return self.full_connection(x4), map