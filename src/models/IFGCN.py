import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
from models.FGCN import FGCN,FGCN_Body
from models.SAGE import SAGE,SAGE_Body
import torch
import torch.nn.functional as F

def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "FGCN":
        model = FGCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    elif args.model == "SAGE":
        model = SAGE_Body(nfeat=nfeat,nhid=args.num_hidden,dropout=args.dropout)
    else:
        print("Model not implement")
        return

    return model

class IFGCN(nn.Module):

    def __init__(self, nfeat, args):
        super(IFGCN,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.estimator = GCN(nfeat,args.hidden,1,dropout)
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,1)


        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)


        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.G_loss = 0

    def forward(self,g,x):
        s = self.estimator(g,x)
        if self.args.model == "FGCN":
            z , _ = self.GNN(g,x)
        else:
            z = self.GNN(g,x)
        y = self.classifier(z)
        return y,s,z
    ##y,s,z 分别是pre，score，x

    def optimize(self,g,x,labels,idx_train,sample_node_pair,input_distance,alpha,beta):
        self.train()

        ### update E, G
        self.optimizer_G.zero_grad()
        if self.args.model == "FGCN":
            h ,out_d= self.GNN(g,x)
            y = self.classifier(h)
            self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())   
            self.ae_loss = F.mse_loss(out_d[idx_train], x[idx_train])
            output_distance = F.cosine_similarity(y[sample_node_pair[0].to(torch.long)], y[sample_node_pair[1].to(torch.long)])
            self.fair_loss = sum(input_distance * output_distance)*0.003
            self.G_loss = alpha* self.cls_loss + beta *self.ae_loss + (1- alpha - beta) * self.fair_loss
        else:
            h = self.GNN(g,x)
            y = self.classifier(h)
            self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())
            self.G_loss = self.cls_loss
                   
        self.G_loss.backward()
        self.optimizer_G.step()




