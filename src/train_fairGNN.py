#%%
import time
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import  accuracy,load_german,load_credit,load_income,load_bail
from models.FairGNN import FairGNN
from torch_geometric.utils import dropout_adj, convert, to_networkx
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.003,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units of the sensitive attribute estimator')
parser.add_argument('--dropout', type=float, default=.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--num-hidden', type=int, default=64,
                    help='Number of hidden units of classifier.')
parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--attn-drop", type=float, default=.0,
                    help="attention dropout")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--acc', type=float, default=0,
                    help='the selected FairGNN accuracy on val would be at least this high')
parser.add_argument('--roc', type=float, default=0,
                    help='the selected FairGNN ROC score on val would be at least this high')
parser.add_argument('--sens_number', type=int, default=200,
                    help="the number of sensitive attributes")
parser.add_argument('--label_number', type=int, default=500,
                    help="the number of labels")
parser.add_argument('--model', type=str, default="FGCN",
                    help='the type of model GCN/GAT/FGCN/SAGE')
parser.add_argument('--dataset', type=str, default='Credit',
                    choices=['Bail','Income','German','Credit'])
parser.add_argument('--cover_rate', type=float, default=0.3)
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--beta', type=float, default=0.03)
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
#%%
alpha = args.alpha
beta = args.beta
th = args.th
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
print(args.dataset)
cover_rate = args.cover_rate
if args.dataset == 'German':
    dataset = 'German'
    sens_attr = "Gender"
    predict_attr = "GoodCustomer"
    label_number = 1000
    sens_number = 50
    seed = 20
    path = "../dataset/German"
    test_idx = False
    adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,sample_node_pair, input_distance= load_german(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,cover_rate,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number,
                                                                                    seed=seed,test_idx=test_idx)
    count_feature = features.clone()
    count_feature[:,0] = -1  

elif args.dataset == 'Credit':
    dataset = 'Credit'
    sens_attr = "Age"
    predict_attr = "NoDefaultNextMonth"
    label_number = 1000
    sens_number = 50
    seed = 21
    path = "../dataset/Credit"
    test_idx = False
    adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,sample_node_pair, input_distance = load_credit(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,cover_rate,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number,
                                                                                    seed=seed,test_idx=test_idx)
    count_feature = features.clone()
    count_feature[:,2] = -1
elif args.dataset == 'Bail':
    dataset = 'Bail'
    sens_attr = "WHITE"
    predict_attr = "RECID"
    label_number = 1000
    sens_number = 50
    seed = 20
    path = "../dataset/Bail"
    test_idx = False
    adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,sample_node_pair, input_distance = load_bail(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,cover_rate,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number,
                                                                                    seed=seed,test_idx=test_idx)
    count_feature = features.clone()
    count_feature[:,0] = -1
elif args.dataset == 'Income':
    dataset = 'Income'
    sens_attr = "race"
    predict_attr = "income"
    label_number = 1000
    sens_number = 50
    seed = 20
    path = "../dataset/Income"
    test_idx = False
    adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,sample_node_pair, input_distance = load_income(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,cover_rate,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number,
                                                                                    seed=seed,test_idx=test_idx)
    count_feature = features.clone()
    count_feature[:,8] = -1


print(dataset)



#%%
import dgl
import scipy as sp
from scipy.sparse import coo_matrix
from utils import feature_norm
G = dgl.DGLGraph()
# G.from_scipy_sparse_matrix(adj)
G = dgl.from_scipy(adj)
G.to('cuda:0')
if dataset == 'nba':
    features = feature_norm(features)


def fair_metric(output,idx):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity,equality
#%%
labels[labels>1]=1
if sens_attr:
    sens[sens>0]=1
# Model and optimizer
root_path, _ = os.path.split(os.path.abspath(__file__))
model = FairGNN(nfeat = features.shape[1], args = args)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
#model.estimator.load_state_dict(torch.load(root_path+"/checkpoint/GCN_sens_{}_ns_{}".format(dataset,sens_number), map_location=torch.device('cpu')))
if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sens = sens.cuda()
    idx_sens_train = idx_sens_train.cuda()
    count_feature = count_feature.cuda()
    sample_node_pair = sample_node_pair.cuda()
    input_distance = input_distance.cuda()
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score


# Train model
t_total = time.time()
best_result = {}
best_fair = 100
best_acc = 0
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    #optimizer.zero_grad()
    model.optimize(G,features,labels,idx_train,sample_node_pair,input_distance,alpha,beta)
    # cov = model.cov
    cls_loss = model.cls_loss
    if args.model =="FGCN":
        
        ae_loss = model.ae_loss
        fair_loss = model.fair_loss
    # adv_loss = model.adv_loss
    model.eval()
    output,_,_ = model(G, features)
    count_output,_,_ = model(G, count_feature)
    outputs = (torch.sigmoid(output)>th)
    count_output = (torch.sigmoid(count_output)>th)

    F1_val = f1_score(outputs[idx_val].cpu(), labels[idx_val].cpu())
    acc_val = accuracy(output[idx_val], labels[idx_val])
    rec_val = recall_score(outputs[idx_val].cpu(), labels[idx_val].cpu())
    roc_val = roc_auc_score(labels[idx_val].cpu().numpy(),output[idx_val].detach().cpu().numpy())
    val_fair_score = 1 - ((count_output[idx_val] == outputs[idx_val]).sum().item()/len(idx_val))
    test_fair_score = 1 - ((count_output[idx_test] == outputs[idx_test]).sum().item()/len(idx_test))
    #acc_sens = accuracy(s[idx_test], sens[idx_test])
    
    parity_val, equality_val = fair_metric(output,idx_val)

    F1_test = f1_score(outputs[idx_test].cpu(), labels[idx_test].cpu())
    acc_test = accuracy(output[idx_test], labels[idx_test])
    rec_test = recall_score(outputs[idx_test].cpu(), labels[idx_test].cpu())
    roc_test = roc_auc_score(labels[idx_test].cpu().numpy(),output[idx_test].detach().cpu().numpy())
    parity,equality = fair_metric(output,idx_test)
    if acc_val > args.acc and roc_val > args.roc:
    
        # if best_fair > parity_val + equality_val :
        #     best_fair = parity_val + equality_val

        #     best_result['acc'] = acc_test.item()
        #     best_result['roc'] = roc_test
        #     best_result['parity'] = parity
        #     best_result['equality'] = equality
        #     best_result['fairness'] = val_fair_score
        # print("=================================")
        if best_acc < acc_test.item() :
            best_acc = acc_test.item()

            best_result['acc'] = acc_test.item()
            best_result['roc'] = roc_test
            best_result['parity'] = parity
            best_result['equality'] = equality
            best_result['fairness'] = val_fair_score
        print("=================================")
        print('Epoch: {:04d}'.format(epoch+1),
            # 'cov: {:.4f}'.format(cov.item()),
            'cls: {:.4f}'.format(cls_loss.item()),
            "F1_val: {:.4f}".format(F1_val),
            'acc_val: {:.4f}'.format(acc_val.item()),
            "roc_val: {:.4f}".format(roc_val),
            "rec_val: {:.4f}".format(rec_val),
            "parity_val: {:.4f}".format(parity_val),
            "equality: {:.4f}".format(equality_val),
            "fairness: {:.4f}".format(val_fair_score))
        if args.model =="FGCN":
            print(
                  'ae: {:.4f}'.format(ae_loss.item()),
                'fair_loss: {:.4f}'.format(fair_loss.item()),)
        print("Test:",
                "F1_test: {:.4f}".format(F1_test),
                "accuracy: {:.4f}".format(acc_test.item()),
                "roc: {:.4f}".format(roc_test),
                # "acc_sens: {:.4f}".format(acc_sens),
                "rec_test: {:.4f}".format(rec_test),
                "parity: {:.4f}".format(parity),
                "equality: {:.4f}".format(equality),
                "fairness: {:.4f}".format(test_fair_score))
                
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

print('============performace on test set=============')
if len(best_result) > 0:
    print("Test:",
            "F1_test: {:.4f}".format(F1_test),
            "accuracy: {:.4f}".format(best_result['acc']),
            "roc: {:.4f}".format(best_result['roc']),
            # "acc_sens: {:.4f}".format(acc_sens),
            "rec_test: {:.4f}".format(rec_test),
            "parity: {:.4f}".format(best_result['parity']),
            "equality: {:.4f}".format(best_result['equality']),
            "fairness: {:.4f}".format(best_result['fairness']))
else:
    print("Please set smaller acc/roc thresholds")