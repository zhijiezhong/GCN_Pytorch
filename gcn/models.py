# -*- coding: utf-8 -*-
# @Time    : 2023/11/5
# @Author  : Zhong Zhijie

import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        """
        初始化图卷积层
        :param nfeat: 输入层维度
        :param nhid: 隐藏层维度
        :param nclass: 节点类别数量
        :param dropout: dropout概率，默认为0.5
        """
        super(GCN, self).__init__()
        # 定义两层图卷积
        self.gcn1 = GraphConvolution(nfeat, nhid)
        self.gcn2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        两层图卷积前向传播
        :param x: 节点的特征矩阵
        :param adj: 预处理后的邻接矩阵
        :return: 经过两层图卷积的结果
        """
        x = F.relu(self.gcn1(x, adj))
        # training=self.training表示的是如果是训练的时候则使用dropout防止过拟合，否则不使用
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn2(x, adj)
        # softmax后使用log（对数）
        return F.log_softmax(x, dim=1)
