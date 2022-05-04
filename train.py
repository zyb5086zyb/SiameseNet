#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： zhaoyoubiao(Fery)
# datetime： 2022/5/4 4:49 下午 
# ide： PyCharm
# --------------------------------
import os
import args
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from simesenet import SiameseNet, lcqmcDataset, ContrastiveLoss

'''1， 创建数据集并建立数据载入器'''
train_data = lcqmcDataset(args.train_path, args.vocab_path)
test_data = lcqmcDataset(args.test_path, args.vocab_path)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

'''2，有gpu使用gpu， 无使用cpu'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SiameseNet().to(device)

'''3， 定义优化方式和损失函数'''
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fun = ContrastiveLoss()

'''4， 训练模型'''
for epoch in range(args.epoch):
    for step, (text_a, text_b, label) in enumerate(train_loader):
        '''4.1载入数据'''
        a = Variable(text_a.to(device).long())
        b = Variable(text_b.to(device).long())
        l = Variable(torch.LongTensor(label).float().to(device))
        '''4.2计算余弦相似度'''
        consine = model(a, b)
        '''4.3预测结果传给loss'''
        loss = loss_fun(consine, l)
        '''4.4 固定格式:
        1)梯度清零
        2)反向传播
        3)梯度迭代'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''4.5 打印日志'''
        if step % 200 == 0:
            total = 0
            correct = 0
            for (test_a, test_b, l) in test_loader:
                test_a = Variable(test_a.to(device).long())
                test_b = Variable(test_b).to(device).long()
                test_l = Variable(torch.LongTensor(l).float().to(device))
                consine = model(test_a, test_b)

                out = torch.Tensor(np.array([1 if cos > args.m else 0 for cos in consine])).long()

                if out.size() == l.size():
                    total += l.size(0)
                    correct += (out==l.cpu()).sum().item()
            print('[EPOCH ~ Step:]', epoch + 1, '~', step + 1, '训练loss:', loss.item())
            print('[EPOCH]:', epoch + 1, '测试准确率:', (correct * 1.0 / total))

torch.save(model, args.model_path)
