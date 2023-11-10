# -*- coding: utf-8 -*-
# @Time    : 2023/11/6
# @Author  : Zhong Zhijie

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='训练的时候不适用cuda')
parser.add_argument('--fastmode', action='store_true', default=False, help='训练的时候并验证')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--epochs', type=int, default=200, help='训练轮次')
parser.add_argument('--lr', type=float, default=0.01, help='学习率')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
parser.add_argument('--hidden', type=int, default=16, help='隐藏层维度大小')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout的概率')

# 获取参数
args = parser.parse_args()
# 如果有cuda且使用cuda，args.cuda才为true
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 设置随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 加载数据集
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# 创建模型
model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
# 优化器Adam
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 如果使用cuda，将一些模型等放到cuda上
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# 训练和验证
def train(epoch):
    # 用于计算训练时间
    t = time.time()
    # 改为训练模型
    model.train()
    # 梯度清零
    optimizer.zero_grad()
    # 前向传播
    output = model(features, adj)
    # 计算训练损失
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # 计算训练准确率
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # 反向传播，计算梯度值
    loss_train.backward()
    # 更新参数
    optimizer.step()

    # fastmode就是不验证
    if not args.fastmode:
        # 改为验证模式
        model.eval()
        # 前向传播
        output = model(features, adj)

    # 计算验证损失
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # 计算验证准确率
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # 打印信息
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


# 测试
def test():
    # 验证模式
    model.eval()
    # 前向传播
    output = model(features, adj)
    # 计算测试损失
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # 计算测试准确率
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # 打印信息
    print('Test set results:',
          'loss_test: {:.4f}'.format(loss_test.item()),
          'acc_test: {:.4f}'.format(acc_test.item()))


if __name__ == '__main__':
    # 用于计算总运行时间
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)

    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))

    test()
