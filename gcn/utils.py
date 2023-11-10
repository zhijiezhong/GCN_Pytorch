# -*- coding: utf-8 -*-
# @Time    : 2023/11/5
# @Author  : Zhong Zhijie

import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    """
    将一维数组转为二维的独热编码
    :param labels: 一维的标签数组
    :return: 二维的独热编码
    """
    # 去重且自动从小到大排序
    classes = set(labels)
    # np.identity(len(classes))是长度为len(classes)的单位矩阵
    # 转为字典的形式，一个数字对应一条向量，比如1对应[0, 1, 0, ...]
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # 再按照labels的顺序映射为对应的向量
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path='../data/cora/', dataset='cora'):
    """
    加载数据集
    :param path: 数据集路径，哪个文件调用这个函数，就看哪个文件的目录
    :param dataset: 数据集名称
    :return:
    """
    print('Loading {} dataset...'.format(dataset))

    # cora.content中的内容就是 索引 特征 标签
    # 现在是转为二维数组，列表中的内容是字符串，维度为(2708, 1435)，节点数量为2708，特征维度为1433
    idx_features_labels = np.genfromtxt('{}{}.content'.format(path, dataset), dtype=np.dtype(str))
    # 稀疏矩阵的形式存储特征矩阵，不明白的可以去了解一下csr,csc和coo三种压缩方法
    features = sp.csr_matrix(idx_features_labels[:, 1: -1], dtype=np.float32)
    # 将标签映射为onehot形式
    labels = encode_onehot(idx_features_labels[:, -1])

    # 获取索引
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    # 建立字典映射idx
    idx_map = {j: i for i, j in enumerate(idx)}
    # 获取相连的节点，也就是边，维度是(5429, 2)
    edges_unordered = np.genfromtxt('{}{}.cites'.format(path, dataset), dtype=np.int32)
    # 先flatten，即打平，变成一维的，然后在将idx映射成连续的，最后reshape成(5429, 2)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # 构建coo形式的稀疏矩阵，传入(data, (row, col)), shape, dtype
    # 这里的data都是1，(row, col)就是一个坐标，shape是N*N，N为节点的数量
    # 这里的adj还只是有向图的邻接矩阵，需要转为无向图的邻接矩阵，即是对称的
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # 构建对称的邻接矩阵，看不懂
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 对特征做归一化，这里的特征矩阵是稀疏的
    features = normalize(features)
    # 传入稀疏矩阵的(A+I)， 结果是D^-1 * (A+I)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # 结果是D^-0.5 * (A+I) * D^-0.5
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 将稀疏的特征矩阵转为numpy格式后转为普通的tensor
    features = torch.FloatTensor(np.array(features.todense()))
    # 转为一维数组
    labels = torch.LongTensor(np.where(labels)[1])
    # 转为torch的稀疏矩阵
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """
    对输入的稀疏矩阵做行归一化，这个输入的稀疏矩阵可以是节点的特征矩阵或者是邻接矩阵
    如果是邻接矩阵，归一化则是D^-1 * (A + I)
    :param mx: 稀疏矩阵，节点的特征矩阵或者是邻接矩阵(A + I)
    :return: 行归一化的矩阵
    """
    # 对矩阵的每一行求和，以邻接矩阵为例子，rowsum即是节点的度D， (N, 1)
    rowsum = np.array(mx.sum(1))
    # r_inv是1/D，(N, 1) -> (N)
    r_inv = np.power(rowsum, -1).flatten()
    # 1/D中如果有为无穷大的，则改为0
    # 不过如果1/D为无穷大，则D为0，若是邻接矩阵是(A + I)，则不会出现这种情况
    r_inv[np.isinf(r_inv)] = 0.
    # 将r_inv转为稀疏的对角矩阵，(N) -> (N, N)
    r_mat_inv = sp.diags(r_inv)
    # 两个矩阵相乘，做行归一化，(N, N) * (N, N) -> (N, N)
    # 如果是特征矩阵，则是(N, N) * (N, d) -> (N, d)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """
    对输入的邻接矩阵做行列归一化
    :param adj: 邻接矩阵(A + I)
    :return: 行列归一化后邻接矩阵
    """
    # 对矩阵每一行求和
    rowsum = np.array(adj.sum(1))
    # D^(-0.5)
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    # 行列归一化，D^(-0.5) * A * D^(-0.5)
    norm_adj = r_mat_inv.dot(adj).dot(r_mat_inv)
    return norm_adj


def accuracy(output, labels):
    """
    计算准确率
    :param output: 经过模型后的输出张量，维度为(N, nclass)
    :param labels: 标签，维度为(N, 1)
    :return: 准确率
    """
    # output.max(1)返回的是output每一行的最大值和对应的索引，维度为(N, 2)
    # output.max(1)[1]则是取出最大值的索引，而不需要最大值，维度为(N, 1)
    # 其实output.max(1)[1]等价于output.argmax(1)
    # type_as则是将类型转为和labels一样的类型
    preds = output.max(1)[1].type_as(labels)
    # 预测值和标签相等则为1.0，否则为0.0
    correct = preds.eq(labels).double()
    # 将正确的个数求和
    correct = correct.sum()
    # 返回正确的个数 / 总个数
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy的稀疏矩阵转为torch的稀疏矩阵
    :param sparse_mx: scipy的稀疏矩阵
    :return: torch的稀疏矩阵
    """
    # 将稀疏矩阵转换为 Coordinate 格式，这种格式将非零元素的行列索引以及对应的数值分别保存在三个分开的数组中。
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # sparse_mx.row和sparse_mx.col分别代表稀疏矩阵非0的元素的行和列，维度均为(1, n)，n为非0元素的数量
    # vstack代表垂直堆叠，堆叠后维度为(2, n)，再转为tensor
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 将非0元素的数据转为tensor
    values = torch.from_numpy(sparse_mx.data)
    # 获取矩阵的大小
    shape = torch.Size(sparse_mx.shape)
    # 构造torch的稀疏矩阵
    return torch.sparse.FloatTensor(indices, values, shape)
