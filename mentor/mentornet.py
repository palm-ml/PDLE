import torch
import torch.nn as nn
import numpy as np 
import math
import torch.nn.functional as F 
from torch.autograd import Variable


class MentorNet(nn.Module):
    def __init__(self, num_classes, num_epochs = 100, label_embedding_size=2, epoch_embedding_size=5, num_fc_nodes=20):
        super(MentorNet, self).__init__()
        if torch.cuda.is_available():
            self.label_embedding = Variable(torch.Tensor(num_classes, label_embedding_size), requires_grad = True).cuda()
            self.epoch_embedding = Variable(torch.Tensor(num_epochs, epoch_embedding_size), requires_grad = True).cuda()
        else:
            self.label_embedding = Variable(torch.Tensor(num_classes, label_embedding_size), requires_grad = True)
            self.epoch_embedding = Variable(torch.Tensor(num_epochs, epoch_embedding_size), requires_grad = True)
        self.init_embedding()
        self.lstm = nn.LSTM(input_size = 2, hidden_size=1, num_layers=1, bias=False, bidirectional=True)
        self.fc1 = nn.Linear(in_features = 2 + label_embedding_size + epoch_embedding_size, out_features = num_fc_nodes)
        self.fc2 = nn.Linear(in_features = num_fc_nodes, out_features = 1)

    def init_hidden(self, batch_size, n_hidden=1):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(2, batch_size, n_hidden, dtype = torch.float)).cuda()
            c0 = Variable(torch.zeros(2, batch_size, n_hidden, dtype = torch.float)).cuda()
        else:
            h0 = Variable(torch.zeros(2, batch_size, n_hidden, dtype = torch.float))
            c0 = Variable(torch.zeros(2, batch_size, n_hidden, dtype = torch.float))
        return (h0, c0)

    def init_embedding(self):
        n1 = self.label_embedding.size(0)*self.label_embedding.size(1)
        self.label_embedding.data.normal_(0, math.sqrt(2. / n1))
        n2 = self.epoch_embedding.size(0)*self.epoch_embedding.size(1)
        self.epoch_embedding.data.normal_(0, math.sqrt(2. / n1))


    def forward(self, in_features):
        """
        Args:
        input_features: a [batch_size, 4] tensor. Each dimension corresponds to
        0: loss, 1: loss difference to the moving average, 2: label and 3: epoch,
        where epoch is an integer between 0 and 99 (the first and the last epoch).
        label_embedding_size: the embedding size for the label feature.
        epoch_embedding_size: the embedding size for the epoch feature.
        num_fc_nodes: number of hidden nodes in the fc layer.
        """
        # 现将特征的每一维度拆解拼接并编码
        batch_size = in_features.size(0)
        # label
        labels = in_features[:,2].view(-1,1).squeeze().long()
        labels_emb = self.label_embedding[labels]
        # epoch
        epochs = in_features[:,3].view(-1,1).squeeze().long()
        # epochs = torch.min(epochs, torch.ones(batch_size,dtype=torch.long)*99)
        epochs_emd = self.epoch_embedding[epochs]
        # loss & loss diff
        losses = in_features[:,0].view(-1,1)
        loss_diffs = in_features[:,1].view(-1,1)
        lstm_inputs = torch.cat((losses, loss_diffs),-1).squeeze().unsqueeze(0)
        hidden0 = self.init_hidden(batch_size)
        loss_emd = self.lstm(lstm_inputs, hidden0)[-1][-1]
        fw_emd = loss_emd[0,:]
        bw_emd = loss_emd[1,:]
        loss_emd = torch.cat((fw_emd, bw_emd), dim=-1)
        # 拼接特征
        if torch.cuda.is_available():
            labels_emb = labels_emb.cuda()
            epochs_emd = epochs_emd.cuda()
        enc_features = torch.cat([labels_emb, epochs_emd, loss_emd],dim=-1)
        v = F.tanh(self.fc1(enc_features))
        return F.sigmoid(self.fc2(v))


if __name__ == '__main__':
    num_classes = 10
    batch_size = 8
    mentor = MentorNet(num_classes)
    inputs = torch.rand(batch_size, 4)
    outputs = mentor(inputs)
    print(outputs)