import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from dgl.nn.pytorch import GraphConv

class FGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(FGCN, self).__init__()
        self.body = FGCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid,nclass)

    def forward(self, g, x):
        x = self.body(g,x)
        x = self.fc(x)
        return x

# def GCN(nn.Module):
class FGCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(FGCN_Body, self).__init__()

        self.conv1 = GraphConv(nfeat, nhid)
        self.conv2 = GraphConv(nhid, nhid)
        self.dec1 = Linear(nhid, nhid)
        self.dec2 = Linear(nhid, nfeat)
        self.mlp1 = Linear(nhid, nhid)
        # self.mlp2 = Linear(nhid , 1)

    def forward(self, g, x):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(g.to('cuda:0'),x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(g.to('cuda:0'),x).relu()

        x_d = F.relu(self.dec1(x))
        x_d = self.dec2(x_d)
        # x = self.mlp1(x)
        # pre = self.mlp2(F.relu(self.mlp1(x)))


        # x = F.relu(self.gc1(g.to('cuda:0'), x))
        # x = self.dropout(x)
        # x = self.gc2(g.to('cuda:0'), x)
        # x = self.dropout(x)
        return x, x_d 




