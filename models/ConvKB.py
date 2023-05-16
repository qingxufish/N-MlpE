import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


class ConvKB(nn.Module):
    def __init__(self, n_ent, n_rel, depth, n_filter, kernel, reg, p):
        super(ConvKB, self).__init__()

        self.depth = depth
        self.n_filter = n_filter
        self.reg = reg
        self.ent_embedding = nn.Embedding(n_ent, depth)
        self.rel_embedding = nn.Embedding(n_rel, depth)
        self.conv1_bn = nn.BatchNorm1d(3)
        self.conv_layer = nn.Conv1d(3, n_filter, kernel_size=kernel)
        self.conv2_bn = nn.BatchNorm1d(n_filter)
        self.dropout = nn.Dropout(p)
        self.w = nn.Linear((depth - kernel + 1) * n_filter, 1, bias=False)
        self.all_params = [self.ent_embedding, self.rel_embedding, self.conv_layer, self.w]

    def initialize(self):
        nn.init.xavier_normal_(self.ent_embedding)
        nn.init.xavier_normal_(self.rel_embedding)
        nn.init.xavier_normal_(self.conv_layer)
        nn.init.xavier_normal_(self.w)

        # self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, dim=1)
        self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, dim=1)

    def get_score(self, heads, tails, rels, clamp=True):
        batch_size = len(heads)
        h = self.ent_embedding(heads)
        t = self.ent_embedding(tails)
        r = self.rel_embedding(rels)

        # inp: .shape: (batch_size, 3, depth)
        inp = torch.cat([h, t, r], dim=1).reshape(batch_size, 3, self.depth)
        inp = self.conv1_bn(inp)
        # out: .shape: (batch_size, n_filter, depth)
        out = self.conv_layer(inp)
        out = self.conv2_bn(out)
        out = F.relu(out)
        # out: .shape: (batch_size, n_filter * depth)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        # out: .shape: (batch_size, 1)
        score = self.w(out).view(-1)
        if clamp:
            score = torch.clamp(score, -20, 20)

        # return shape: (batch_size,)
        return score

    def forward(self, x, labels):
        self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, dim=1)

        # shape: (batch_size,)
        heads, tails, rels = x[:, 0], x[:, 1], x[:, 2]
        scores = self.get_score(heads, tails, rels)

        if self.reg == 0.:
            return F.softplus(labels * scores).mean()
        return F.softplus(labels * scores).mean() + self.reg * self.get_regularization()

    def get_regularization(self):
        penalty = 0
        for param in self.all_params:
            penalty += torch.sum(param.weight.data ** 2) / 2.
        return penalty