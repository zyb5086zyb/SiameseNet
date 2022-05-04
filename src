import torch
from torch.utils.data import DataLoader,Dataset
from data_load import get_embed
from torch.autograd import Variable
from train import LcqmcDataset
import args


if __name__ == '__main__':
    embed, char2idx, idx2char = get_embed(args.VOCAB_FILE)

    test_data = LcqmcDataset(args.TEST_DATA,args.VOCAB_FILE)
    test_loader=DataLoader(dataset=test_data,batch_size=args.BATCH_SIZE,shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net=torch.load(args.MODEL_FILE).to(device)

    total = 0
    correct = 0
    TP,TN,FP,FN=0,0,0,0
    FLAG=True
    for (test_a, test_b, test_l) in test_loader:
        tst_a = Variable(test_a.to(device).long())
        tst_b = Variable(test_b.to(device).long())
        tst_l = Variable(torch.LongTensor(test_l).to(device))

        pos_res = net(tst_a, tst_b)
        neg_res = 1 - pos_res
        out = torch.max(torch.stack([neg_res, pos_res], 1).to(device), dim=1)[1]

        total += tst_l.size(0)
        correct += (out == tst_l).sum().item()

        TP += ((out == 1) & (tst_l == 1)).sum().item()
        TN += ((out == 0) & (tst_l == 0)).sum().item()
        FN += ((out == 0) & (tst_l == 1)).sum().item()
        FP += ((out == 1) & (tst_l == 0)).sum().item()

        if FLAG == True:
           for i in range(10,20):
               a, b, l = test_data[i][0], test_data[i][1], test_data[i][2]
               print('句子A：',end='')
               for id in a:
                   if id==0:
                       break;
                   print(idx2char[id],end='')
               print('\n句子B：',end='')
               for id in b:
                   if id==0:
                       break;
                   print(idx2char[id],end='')
               print('\n标签：',l,'预测：',out[i].item())

           FLAG=False

    p = TP / (TP + FP)
    r = TP / (TP + FN)

    print('测试集准确率: ', (correct * 1.0 / total))
    print('测试集精确率：', p)
    print('测试集召回率：', r)
    print('测试集f1-score：', 2 * r * p / (r + p))
