import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import SiameseNet,LcqmcDataset,ContrastiveLoss
from data_load import get_embed
import numpy as np
import args


if __name__ == '__main__':
    embed, char2idx, idx2char = get_embed(args.VOCAB_FILE)

    train_data=LcqmcDataset(args.TRAIN_DATA,args.VOCAB_FILE)
    test_data=LcqmcDataset(args.TEST_DATA,args.VOCAB_FILE)
    train_loader=DataLoader(dataset=train_data,batch_size=args.BATCH_SIZE,shuffle=True)
    test_loader=DataLoader(dataset=test_data,batch_size=args.BATCH_SIZE,shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'

    net=SiameseNet(embed).to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=args.LR)
    loss_func = ContrastiveLoss()

    for epoch in range(args.EPOCH):
        for step, (text_a, text_b, label) in enumerate(train_loader):
            # 1、把索引转化为tensor变量，载入设备，注意转化成long tensor
            a = Variable(text_a.to(device).long())
            b = Variable(text_b.to(device).long())
            l = Variable(torch.LongTensor(label).float().to(device))

            # 2、计算余弦相似度
            cosine = net(a, b)

            # 3、预测结果传给loss
            loss = loss_func(cosine, l)

            # 4、固定格式
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 200 == 0:
                total = 0
                correct = 0
                for (test_a, test_b, test_l) in test_loader:
                    tst_a = Variable(test_a.to(device).long())
                    tst_b = Variable(test_b.to(device).long())
                    tst_l = Variable(torch.LongTensor(test_l).to(device))
                    cosine = net(tst_a, tst_b)
                    #     print(cosine)
                    out = torch.Tensor(np.array([1 if cos > args.M else 0 for cos in cosine])).long()
                    #     print(out)
                    if out.size() == tst_l.size():
                        total += tst_l.size(0)
                        correct += (out == tst_l.cpu()).sum().item()
                print('[Epoch ~ Step]:', epoch + 1, '~', step + 1, '训练loss:', loss.item())
                print('[Epoch]:', epoch + 1, '测试集准确率: ', (correct * 1.0 / total))

    torch.save(net, args.MODEL_FILE)
