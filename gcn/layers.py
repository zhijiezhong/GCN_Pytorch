# -*- coding: utf-8 -*-
# @Time    : 2023/11/5
# @Author  : Zhong Zhijie

import math
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        初始化一些必要的变量
        :param in_features: 输入维度
        :param out_features: 输出维度
        :param bias: 是否需要偏置值，默认为True
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 定义一个可训练的参数矩阵W
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            # 定义偏置值
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # stdv是standard deviation，即标准差
        # self.weight.size(1)是输出维度的大小，即out_features
        stdv = 1. / math.sqrt(self.weight.size(1))
        # 均匀分布随机初始化W
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # 均匀分布随机初始化b
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        一层图卷积的前向传播，三个矩阵相乘：A(input)W
        :param input: 如果是第一层，则input=X，是节点的特征矩阵
        :param adj: 预处理后的邻接矩阵
        :return: ：A(input)W
        """
        # 这里是input*W，其实这个操作就是线性层的实现，和直接使用Linear是一样的
        support = torch.mm(input, self.weight)
        # 邻接矩阵和上面获取到的结果相乘，A(input)W，只不过这里使用到稀疏矩阵乘法，可以减少内存
        output = torch.spmm(adj, support)
        # 如果需要偏置值，加上bias
        if self.bias is not None:
            output += self.bias
        return output

    # 这个函数的作用：当在打印这个类的对象时，会输出下面的字符串，类似于java中的toString方法
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
