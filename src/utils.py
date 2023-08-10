#%%
import numpy as np
import scipy.sparse as sp
import torch
import os
import pandas as pd
import dgl
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
import torch.nn.functional as F
#%%

#%%


def load_german(dataset,sens_attr,predict_attr, cover_rate,path="/dataset/German/", label_number=1000,sens_number=500,seed=19,test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))
    root_path, _ = os.path.split(os.path.abspath(__file__))

    idx_features_labels = pd.read_csv(root_path[:-4]+'/dataset/German/{}.csv'.format(dataset))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')
    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0


    # build graph
    
    edges_unordered = np.loadtxt(root_path[:-4]+'/Dataset/{}/{}_edges.txt'.format(dataset, dataset), dtype=int).T
    edge_list = torch.LongTensor(edges_unordered)
    node_list = edge_list.unique()
    node_len = len(node_list)

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    features = 2*((features - features.min(dim=0)[0]) / (features.max(dim=0)[0] - features.min(dim=0)[0]))-1
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]




    sens = idx_features_labels[sens_attr].values.astype(int)

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    # 删特征
    tmp_node_list = node_list.tolist()
    tmp_node_list = torch.tensor(random.sample(tmp_node_list, k=int(len(tmp_node_list) * cover_rate)))
    
    features[tmp_node_list][:,0] = -1

    features = features.float()
    # 相似度点对
    train_len = 0.5 * len(label_idx)
    train_node = label_idx[:int(train_len)].tolist()
    tmp = int(train_len * 0.2)
    if(tmp % 2 != 0):
        tmp += 1
    sample_node_pair = torch.tensor(np.array(random.sample(train_node, k=tmp)))
    sample_node_pair = sample_node_pair.reshape(2, sample_node_pair.shape[0] // 2)

    input_distance = F.cosine_similarity(features[sample_node_pair[0].to(torch.long)], features[sample_node_pair[1].to(torch.long)])


    # random.shuffle(sens_idx)

    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train ,sample_node_pair,input_distance

def load_income(dataset,sens_attr,predict_attr, cover_rate, path="/dataset/Credit/", label_number=1000,sens_number=500,seed=20,test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))
    root_path, _ = os.path.split(os.path.abspath(__file__))

    idx_features_labels = pd.read_csv(root_path[:-4]+'/dataset/Income/{}.csv'.format(dataset))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    # Sensitive Attribute

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0


    # build graph
    edges_unordered = np.loadtxt(root_path[:-4]+'/Dataset/{}/{}_edges.txt'.format(dataset, dataset), dtype=int).T
    edge_list = torch.LongTensor(edges_unordered)
    node_list = edge_list.unique()
    node_len = len(node_list)


    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    features = 2*((features - features.min(dim=0)[0]) / (features.max(dim=0)[0] - features.min(dim=0)[0]))-1
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values.astype(int)

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    

    # 删特征
    tmp_node_list = node_list.tolist()
    tmp_node_list = torch.tensor(random.sample(tmp_node_list, k=int(len(tmp_node_list) * cover_rate)))
    
    features[tmp_node_list][:,8] = -1

    features = features.float()
    # 相似度点对
    train_len = 0.5 * len(label_idx)
    train_node = label_idx[:int(train_len)].tolist()
    tmp = int(train_len * 0.2)
    if(tmp % 2 != 0):
        tmp += 1
    sample_node_pair = torch.tensor(np.array(random.sample(train_node, k=tmp)))
    sample_node_pair = sample_node_pair.reshape(2, sample_node_pair.shape[0] // 2)

    input_distance = F.cosine_similarity(features[sample_node_pair[0].to(torch.long)], features[sample_node_pair[1].to(torch.long)])

    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train,sample_node_pair,input_distance
def load_bail(dataset,sens_attr,predict_attr, cover_rate, path="/dataset/Bail/", label_number=1000,sens_number=500,seed=20,test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))
    root_path, _ = os.path.split(os.path.abspath(__file__))

    idx_features_labels = pd.read_csv(root_path[:-4]+'/dataset/Bail/{}.csv'.format(dataset))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    # Sensitive Attribute

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0


    # build graph
    edges_unordered = np.loadtxt(root_path[:-4]+'/Dataset/{}/{}_edges.txt'.format(dataset, dataset), dtype=int).T
    edge_list = torch.LongTensor(edges_unordered)
    node_list = edge_list.unique()
    node_len = len(node_list)


    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    features = 2*((features - features.min(dim=0)[0]) / (features.max(dim=0)[0] - features.min(dim=0)[0]))-1
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values.astype(int)

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    

    # 删特征
    tmp_node_list = node_list.tolist()
    tmp_node_list = torch.tensor(random.sample(tmp_node_list, k=int(len(tmp_node_list) * cover_rate)))
    
    features[tmp_node_list][:,0] = -1

    features = features.float()
    # 相似度点对
    train_len = 0.5 * len(label_idx)
    train_node = label_idx[:int(train_len)].tolist()
    tmp = int(train_len * 0.2)
    if(tmp % 2 != 0):
        tmp += 1
    sample_node_pair = torch.tensor(np.array(random.sample(train_node, k=tmp)))
    sample_node_pair = sample_node_pair.reshape(2, sample_node_pair.shape[0] // 2)

    input_distance = F.cosine_similarity(features[sample_node_pair[0].to(torch.long)], features[sample_node_pair[1].to(torch.long)])

    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train,sample_node_pair,input_distance
def load_credit(dataset,sens_attr,predict_attr, cover_rate, path="/dataset/Credit/", label_number=1000,sens_number=500,seed=19,test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))
    root_path, _ = os.path.split(os.path.abspath(__file__))

    idx_features_labels = pd.read_csv(root_path[:-4]+'/dataset/Credit/{}.csv'.format(dataset))
    header = list(idx_features_labels.columns)
    count_header = header
    header.remove(predict_attr)

    # Sensitive Attribute

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    count_feature =sp.csr_matrix(idx_features_labels[count_header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0


    # build graph
    edges_unordered = np.loadtxt(root_path[:-4]+'/Dataset/{}/{}_edges.txt'.format(dataset, dataset), dtype=int).T
    edge_list = torch.LongTensor(edges_unordered)
    node_list = edge_list.unique()
    node_len = len(node_list)


    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    features = 2*((features - features.min(dim=0)[0]) / (features.max(dim=0)[0] - features.min(dim=0)[0]))-1
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]




    sens = idx_features_labels[sens_attr].values.astype(int)

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    

    # 删特征
    tmp_node_list = node_list.tolist()
    tmp_node_list = torch.tensor(random.sample(tmp_node_list, k=int(len(tmp_node_list) * cover_rate)))
    
    features[tmp_node_list][:,2] = -1

    features = features.float()
    # 相似度点对
    train_len = 0.5 * len(label_idx)
    train_node = label_idx[:int(train_len)].tolist()
    tmp = int(train_len * 0.2)
    if(tmp % 2 != 0):
        tmp += 1
    sample_node_pair = torch.tensor(np.array(random.sample(train_node, k=tmp)))
    sample_node_pair = sample_node_pair.reshape(2, sample_node_pair.shape[0] // 2)

    input_distance = F.cosine_similarity(features[sample_node_pair[0].to(torch.long)], features[sample_node_pair[1].to(torch.long)])

    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train,sample_node_pair,input_distance

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_norm(features):

    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return 2*(features - min_values).div(max_values-min_values) - 1

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#%%



#%%
