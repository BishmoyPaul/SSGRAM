import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphFeatureProcessor(nn.Module):
    def __init__(self, band, feature, alpha, beta, return_attn_map = False, correlation = 'dot'):
        super(GraphFeatureProcessor, self).__init__()

        self.band = band
        self.feature = feature

        self.Wk = nn.Parameter(torch.empty(size=(1, band, feature)))
        self.Wv = nn.Parameter(torch.empty(size=(1, band, feature)))
        self.Wn = nn.Parameter(torch.empty(size=(1, feature, feature)))

        nn.init.xavier_uniform_(self.Wk.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wv.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wn.data, gain=1.414)

        self.alpha = alpha
        self.beta = beta
        self.LRelu = nn.LeakyReLU(self.alpha)
        self.return_attn_map = return_attn_map

        self.correlation = correlation
        if self.correlation == 'concatenate':
              self.W4 = nn.Parameter(torch.empty(size=(1, feature*2, feature)))
              nn.init.xavier_uniform_(self.W4.data, gain=1.414)

        elif self.correlation == 'cosine':
           self.cosine = nn.CosineSimilarity(dim=-1, eps=1e-6)



    def forward(self, x):

        bs, h, w,_ = x.shape
        ch, cw = h//2, w//2

        h1 = self.LRelu(torch.matmul(x, self.Wk))
        h2 = self.LRelu(torch.matmul(x, self.Wv))

        central1 = h1[:,ch,cw,:].view(bs,1,1,-1)
        if self.correlation == 'dot':
          corr = torch.mul(h1,central1)
          mean_corr = torch.mean(corr,-1)

        elif self.correlation == 'concatenate':
          combined_matrix = torch.cat([h1, central1.expand((bs,h,w,-1))],-1)
          corr = torch.matmul(combined_matrix, self.W4)
          mean_corr = torch.mean(corr,-1)

        elif self.correlation == 'cosine':
          mean_corr = self.cosine(h1, central1)

        central2 = h2[:,ch,cw,:].view(bs,-1)

        attn_map = F.softmax(mean_corr)
        neighbour_feat = torch.sum(torch.mul(h2,attn_map.unsqueeze(-1)).view(bs,-1,self.feature),1)

        central_feat = central2
        central_feat = self.LRelu(torch.matmul(self.Wn,central2.unsqueeze(-1))).squeeze(-1)

        total_feat = self.beta*central_feat + (1-self.beta)*neighbour_feat
        if self.return_attn_map:
          return total_feat, attn_map
        return  total_feat



class GraphAttentionFeatureProcessor(nn.Module):
    def __init__(self,band, feature, central_size = 3, gamma = 0.7,
                 residual = True, get_attn_map = False,
                 correlation = "dot", fc_classes = -1):
        super(GraphAttentionFeatureProcessor, self).__init__()

        self.name = 'GAFP'
        self.band = band
        self.feature = feature

        self.central_size = central_size
        self.left_idx = central_size//2
        self.right_idx = central_size - self.left_idx

        self.gamma = gamma

        self.residual = residual
        self.get_attn_map = get_attn_map
        self.downsampler = nn.AvgPool2d(kernel_size = (3,3), stride=2, padding=0)

        self.graph_processor = GraphFeatureProcessor
        feature = self.feature


        self.neighbour = self.graph_processor(band, feature, 0.2, 0.7, return_attn_map = get_attn_map, correlation = correlation)
        self.core = self.graph_processor(band, feature, 0.2, 0.7,correlation=correlation)

        self.fc_classes = fc_classes
        if fc_classes > 0:
          self.fc = nn.Linear(feature, fc_classes)



    def forward(self, x):

        x = x.squeeze(1)
        bs, h, w,_ = x.shape
        ch, cw = h//2, w//2
        central_patch = x[:,ch-self.left_idx:ch+self.right_idx,cw-self.left_idx:cw+self.right_idx,:]


        combined_patch = x
        core_features = self.core(central_patch)

        if not self.get_attn_map:
          neighbour_features = self.neighbour(combined_patch)
          total_feat = core_features*self.gamma + (1 - self.gamma)*neighbour_features

          if self.fc_classes > 0:
            total_feat = self.fc(total_feat)
            return  total_feat, None
          return total_feat

        else:

          neighbour_features,attn_map = self.neighbour(combined_patch)
          attn_map = attn_map.unsqueeze(1)

          total_feat = core_features*self.gamma + (1 - self.gamma)*neighbour_features

          if self.fc_classes > 0:
            total_feat = self.fc(total_feat)
            return total_feat, attn_map

          return total_feat,attn_map