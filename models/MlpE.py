import torch
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel


class MlpE(BaseModel):
    def __init__(self, config):
        super(MlpE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.entity_dim = kwargs.get('entity_dim')
        self.relation_dim = kwargs.get('relation_dim')
        self.E = torch.nn.Embedding(self.entity_cnt, self.entity_dim)
        self.R = torch.nn.Embedding(self.relation_cnt * 2, self.relation_dim)
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.feature_drop = torch.nn.Dropout(kwargs.get('feature_dropout'))

        self.bn0 = torch.nn.BatchNorm1d(1)  # batch normalization over a 4D input
        self.bn2 = torch.nn.BatchNorm1d(1)
        # self.register_parameter('b', Parameter(torch.zeros(self.entity_cnt)))
        self.register_parameter('c', Parameter(torch.ones(1)))

        self.encoder = torch.nn.Linear(self.entity_dim+self.relation_dim, self.entity_dim*self.relation_dim)
        self.decoder = torch.nn.Linear(self.entity_dim*self.relation_dim, self.entity_dim)
        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector / scalar
        return unitized_vector

    def forward(self, batch_h, batch_r, batch_t=None):
        # (h,r,t)
        batch_size = batch_h.size(0)
        e1 = self.unitized(self.E(batch_h))
        r = self.unitized(self.R(batch_r))
        stacked_inputs = torch.cat([e1, r], 1).view(batch_size, 1, -1)  # (batch_size, 1, 2*emb_dim1, emb_dim2)

        x = self.bn0(stacked_inputs)
        x = self.encoder(x)
        x = self.feature_drop(x)
        x = torch.relu(x)
        x = self.decoder(x)
        x = self.bn2(x)
        x = self.hidden_drop(x).view(batch_size, -1)
        x = self.unitized(x) * self.c
        # x = F.relu(x)  # deletable
        entities_embedding = self.unitized(self.E.weight)
        x = torch.mm(x, entities_embedding.transpose(1, 0))  # *self.c  # c is important
        # x += self.b.expand_as(x)  # deletable
        y = torch.sigmoid(x)

        return self.loss(y, batch_t), y


class ConvELoss(BaseModel):
    def __init__(self, device, label_smoothing, entity_cnt):
        super().__init__()
        self.device = device
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.label_smoothing = label_smoothing
        self.entity_cnt = entity_cnt

    def forward(self, batch_p, batch_t=None):
        batch_size = batch_p.shape[0]
        loss = None
        if batch_t is not None:
            batch_e = torch.zeros(batch_size, self.entity_cnt).to(self.device).scatter_(1, batch_t.view(-1, 1), 1)
            batch_e = (1.0 - self.label_smoothing) * batch_e + self.label_smoothing / self.entity_cnt
            loss = self.loss(batch_p, batch_e) / batch_size
            return loss