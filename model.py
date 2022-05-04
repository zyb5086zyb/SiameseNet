import torch
import torch.nn as nn
import args
from data_load import load_char_data
from torch.utils.data import Dataset


class SiameseNet(torch.nn.Module):
    def __init__(self,embed):
        super(SiameseNet, self).__init__()
        self.embedding = nn.Embedding(args.CHAR_SIZE, args.EMBEDDING_SIZE)
        self.embedding.weight.data.copy_(torch.from_numpy(embed))
        self.bi_lstm = nn.LSTM(args.EMBEDDING_SIZE, args.LSTM_HIDDEN_SIZE, num_layers=2, dropout=0.2,
                               batch_first=True, bidirectional=True)
        self.dense = nn.Linear(args.LINEAR_HIDDEN_SIZE, args.LINEAR_HIDDEN_SIZE)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, a, b):
        emb_a = self.embedding(a)
        emb_b = self.embedding(b)

        lstm_a, (h_a, c_a) = self.bi_lstm(emb_a)
        lstm_b, (h_b, c_b) = self.bi_lstm(emb_b)

        avg_a = torch.mean(lstm_a, dim=1)
        avg_b = torch.mean(lstm_b, dim=1)

        out_a = torch.tanh(self.dense(avg_a))
        out_a = self.dropout(out_a)
        out_b = torch.tanh(self.dense(avg_b))
        out_b = self.dropout(out_b)

        cosine = torch.cosine_similarity(out_a, out_b, dim=1, eps=1e-8)
        return cosine

class LcqmcDataset(Dataset):
    def __init__(self, filepath, vocab_file):
        self.path = filepath
        self.a_index, self.b_index, self.label = load_char_data(filepath, vocab_file)

    def __len__(self):
        return len(self.a_index)

    def __getitem__(self, idx):
        return self.a_index[idx], self.b_index[idx], self.label[idx]


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, Ew, y):
        l_1 = 0.25 * (1.0 - Ew) * (1.0 - Ew)
        l_0 = torch.where(Ew < args.M * torch.ones_like(Ew), torch.full_like(Ew, 0), Ew) * torch.where(
            Ew < args.M * torch.ones_like(Ew), torch.full_like(Ew, 0), Ew)

        loss = y * 1.0 * l_1 + (1 - y) * 1.0 * l_0
        return loss.sum()
