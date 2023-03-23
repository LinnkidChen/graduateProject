import argparse
import os.path as osp
import os
import random
import nni
import yaml
from yaml import SafeLoader
import numpy as np
import scipy
import scipy.sparse as sp
import torch
from torch_scatter import scatter_add
import torch.nn as nn
from torch_geometric.utils import dropout_adj, degree, to_undirected, get_laplacian
import torch.nn.functional as F
import networkx as nx
from scipy.sparse.linalg import eigs, eigsh
from pytorchtools import EarlyStopping
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from simple_param.sp import SimpleParam
from itertools import product
from pGRACE.model import Encoder, GRACE, NewGConv, NewEncoder, NewGRACE
from pGRACE.functional import (
    drop_feature,
    drop_edge_weighted,
    degree_drop_weights,
    evc_drop_weights,
    pr_drop_weights,
    feature_drop_weights,
    drop_feature_weighted_2,
    feature_drop_weights_dense,
)
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import (
    get_base_model,
    get_activation,
    generate_split,
    compute_pr,
    eigenvector_centrality,
)
from pGRACE.dataset import get_dataset
from utils import (
    normalize_adjacency_matrix,
    create_adjacency_matrix,
    load_adj_neg,
    Rank,
)


def process_data():
    # 用heco的baseline
    # train_20 train类别每个label有20个节点
    # ACM
    # # feat_path = '/nfs_baoding_ai/xumeng/run_emb/HeCo/data/acm/p_feat.npz'
    # feat_path = "../data/acm/p_feat.npz"
    # # feat_path = "./data/acm/p_feat.npz"  # paper feature (x)
    # # path_1 = '/nfs_baoding_ai/xumeng/run_emb/HeCo/data/acm/pap_idx.npy'
    # path_1 = "../data/acm/pap_idx.npy"
    # # path_1 = "./data/acm/pap_idx.npy"  # metapath pap 临接
    # # path_2 = '/nfs_baoding_ai/xumeng/run_emb/HeCo/data/acm/psp_idx.npy'
    # # path_2 = "./data/acm/psp_idx.npy"  # metapath psp 临接
    # path_2 = "../data/acm/psp_idx.npy"
    # # path = '/nfs_baoding_ai/xumeng/run_emb/HeCo/data/acm/'
    # # path = "./data/acm/"
    # path = "../data/acm/"
    # pap = torch.from_numpy(np.load(path_1))  # paper author paper 将pa ap 两条边合成一条片p-p
    # psp = torch.from_numpy(np.load(path_2))  # paper subject paper

    # Aminer
    feat_path = "/root/graduateProject/21.12.23代码/data/dblp/a_feat.npz"
    path_1 = "/root/graduateProject/21.12.23代码/data/dblp/apa_idx.npy"
    path_2 = "/root/graduateProject/21.12.23代码/data/dblp/apcpa_idx.npy"
    path_3 = "/root/graduateProject/21.12.23代码/data/dblp/aptpa_idx.npy"
    path = "/root/graduateProject/21.12.23代码/data/dblp/"
    # Aminer
    # feat_path = "/Users/tongchen/Library/Mobile Documents/com~apple~CloudDocs/毕业设计/graduateProject/21.12.23代码/data/dblp/a_feat.npz"
    # path_1 = "/Users/tongchen/Library/Mobile Documents/com~apple~CloudDocs/毕业设计/graduateProject/21.12.23代码/data/dblp/apa_idx.npy"
    # path_2 = "/Users/tongchen/Library/Mobile Documents/com~apple~CloudDocs/毕业设计/graduateProject/21.12.23代码/data/dblp/apcpa_idx.npy"
    # path_3 = "/Users/tongchen/Library/Mobile Documents/com~apple~CloudDocs/毕业设计/graduateProject/21.12.23代码/data/dblp/aptpa_idx.npy"
    # path = "/Users/tongchen/Library/Mobile Documents/com~apple~CloudDocs/毕业设计/graduateProject/21.12.23代码/data/dblp/"
    # apa = torch.from_numpy(np.load(path_1))
    # apcpa = torch.from_numpy(np.load(path_2))
    # aptpa = torch.from_numpy(np.load(path_3))
    apa = torch.from_numpy(np.load(path_1)).type(torch.LongTensor)
    apcpa = torch.from_numpy(np.load(path_2)).type(torch.LongTensor)
    aptpa = torch.from_numpy(np.load(path_3)).type(torch.LongTensor)

    # 目的是预测paper的类别
    features = sp.load_npz(feat_path)
    features = torch.from_numpy(features.todense())
    label = np.load(path + "labels.npy").astype("int32")  # Y
    train = torch.from_numpy(np.load(path + "train_" + "20" + ".npy"))
    test = torch.from_numpy(np.load(path + "test_" + "20" + ".npy"))
    val = torch.from_numpy(np.load(path + "val_" + "20" + ".npy"))

    return [apa, apcpa, aptpa], features, label, train, val, test
    # edge_list, features, label, train_idx, val, test_idx


def train():
    model.train()
    optimizer.zero_grad()
    # edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
    # edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0] #adjacency with edge droprate 2
    edge_index_1 = edge_list[0].to(device)  # 邻接矩阵 扰动
    edge_index_2 = edge_list[1].to(device)
    edge_index_3 = edge_list[2].to(device)
    # edge_index_1 = dropout_adj(edge_index_1, p=drop_edge_rate_1)[0]
    # edge_index_2 = dropout_adj(edge_index_2, p=drop_edge_rate_2)[0]
    # x_1 = drop_feature(features, drop_feature_rate_1)#3
    # x_2 = drop_feature(features, drop_feature_rate_2)#4
    x_1 = features  # 特征矩阵 没有扰动
    x_2 = features
    x_3 = features
    z1 = model(
        x_1, edge_index_1, [2, 4]
    )  # a(axW1)W2, --> a^4(a^2xW1)W2 ->GCN(encoder)  W1,W2两层的参数 A=邻接矩阵 X=特征矩阵
    z2 = model(x_2, edge_index_2, [8, 4])
    z3 = model(x_3, edge_index_3, [8, 4])
    loss = model.loss(  # GRACE infoNce
        z1,
        z2,
        z3,
        batch_size=64
        if args.dataset == "Coauthor-Phy" or args.dataset == "ogbn-arxiv"
        else None,
    )
    loss.backward()
    optimizer.step()

    return loss.item()


def test(final=False):

    model.eval()
    z1 = model(features, edge_index_1, [1, 1], final=True)
    z2 = model(features, edge_index_2, [1, 1], final=True)
    z3 = model(features, edge_index_3, [1, 1], final=True)
    z = z1 + z2 + z3

    evaluator = MulticlassEvaluator()

    acc = log_regression(z, label, split, evaluator, num_epochs=200)["acc"]

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc  # , acc_1, acc_2


def my_train(
    learning_rate,
    num_hidden,
    num_proj_hidden,
    activation,
    base_model,
    num_layers,
    drop_edge_rate_1,
    drop_edge_rate_2,
    drop_feature_rate_1,
    drop_feature_rate_2,
    tau,
    num_epochs,
    weight_decay,
    drop_scheme,
    rand_layers,
):
    tmpstring = " - ".join(
            [
                str(i)
                for i in (
                    learning_rate,
                    num_hidden,
                    num_proj_hidden,
                    activation,
                    base_model,
                    num_layers,
                    drop_edge_rate_1,
                    drop_edge_rate_2,
                    drop_feature_rate_1,
                    drop_feature_rate_2,
                    tau,
                    num_epochs,
                    weight_decay,
                    drop_scheme,
                    rand_layers,
                )
            ]
        )
    print(tmpstring)
    with open("result.yaml") as f:
        res=yaml.load(f,Loader=SafeLoader)
        res=list(res.keys())
        if tmpstring in res:
            return 
    global edge_list, features, label, split
    edge_list, features, label, train_idx, val, test_idx = process_data()
    # edge_list[0].shape=[57853,2]
    # edge_list[1].shape= [4338213,2]
    # features.shape=[4019,1902]
    # label.shape=(4019)
    # train_idx.shape=([60])
    # val.shape=([1000])
    # test_idx.shape=([1000])
    early = EarlyStopping(patience=patience, verbose=True)
    edge_list = [idx.t().to(device) for idx in edge_list]
    # t()=transpose()
    # print("weightdecay", weight_decay)
    features = features.float().to(device)
    split = {"train": train_idx, "val": val, "test": test_idx}
    # if args.dataset == 'Cora' or args.dataset == 'CiteSeer' or  args.dataset == 'PubMed': split = (data.train_mask, data.val_mask, data.test_mask)

    encoder = NewEncoder(
        features.shape[1],
        num_hidden,
        get_activation(activation),
        base_model=NewGConv,
        k=num_layers,
    ).to(device)
    adj = 0
    global model
    model = NewGRACE(encoder, adj, num_hidden, num_proj_hidden, tau).to(
        device
    )  # 对比学习的模型 loss在这个里边
    global optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    log = args.verbose.split(",")
    global edge_index_1, edge_index_2, edge_index_3
    edge_index_1 = edge_list[0]
    edge_index_2 = edge_list[1]
    edge_index_3 = edge_list[2]
    print("running..")

    for epoch in range(1, num_epochs + 1):
        print("start of epoch ", epoch)
        loss = train()
        if "train" in log:
            print(f"(T) | Epoch={epoch:03d}, loss={loss:.4f}")
        if epoch % 10 == 0:
            acc = test()
            if "eval" in log:
                print(f"(E) | Epoch={epoch:04d}, avg_acc = {acc}")
        early(loss, model)
        if early.early_stop:
            print("Early stopping")
            num_epochs=epoch
            break

    acc = test(final=True)
    with open("result.yaml", "a") as f:
        print("printing to resultç")
        
        yaml.dump({tmpstring: {"acc": acc, "loss": loss}}, f)
    if "final" in log:
        print(f"{acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="ACM")
    parser.add_argument("--config", type=str, default="param.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=str, default="train,eval,final")
    parser.add_argument("--save_split", type=str, nargs="?")
    parser.add_argument("--load_split", type=str, nargs="?")
    args = parser.parse_args()
    print(os.getcwd())
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    # config = yaml.load(open("./MAGCL-main/param.yaml"), Loader=SafeLoader)[args.dataset]
    dataset = args.dataset
    device = args.device
    torch.manual_seed(args.seed)
    random.seed(0)
    np.random.seed(args.seed)
    use_nni = args.config == "nni"
    learning_rate = config["learning_rate"]  # para2
    num_hidden = config["num_hidden"]
    num_proj_hidden = config["num_proj_hidden"]
    activation = config["activation"]
    base_model = config["base_model"]
    num_layers = config["num_layers"]
    drop_edge_rate_1 = config["drop_edge_rate_1"]
    drop_edge_rate_2 = config["drop_edge_rate_2"]
    drop_feature_rate_1 = config["drop_feature_rate_1"]
    drop_feature_rate_2 = config["drop_feature_rate_2"]
    drop_scheme = config["drop_scheme"]
    tau = config["tau"]
    num_epochs = 10  # para1
    weight_decay = config["weight_decay"]
    rand_layers = config["rand_layers"]
    patience = config["patience"][0]
    excludes = ["patience"]
    parameter_value = [config[k] for k in config.keys() if k not in excludes]
    for item in product(*parameter_value):
        loss = my_train(*item)

    device = torch.device(args.device)
