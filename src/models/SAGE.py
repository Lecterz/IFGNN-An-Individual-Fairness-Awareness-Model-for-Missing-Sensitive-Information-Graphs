import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SAGE, self).__init__()
        self.body = SAGE_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, g, x):
        x = self.body(g, x)
        x = self.fc(x)
        return x

class SAGE_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(SAGE_Body, self).__init__()

        self.sage1 = SAGEConv(nfeat, nhid, 'mean')  
        self.sage1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU()
#             nn.BatchNorm1d(nhid),
#             nn.Dropout(p=dropout)
        )

    def forward(self, g, x):
        x = self.sage1(g.to('cuda:0'),x)
        x = self.transition(x)

        return x