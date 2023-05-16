import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel


class ConvR(BaseModel):
    def __init__(self, config):
        super(ConvR, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.reshape = kwargs.get('reshape')
        self.kernel_size = kwargs.get('conv_kernel_size')
        self.stride = kwargs.get('stride')
        self.emb_dim = {
            'entity': kwargs.get('emb_dim'),
            'relation': self.conv_out_channels * self.kernel_size[0] * self.kernel_size[1]
        }
        assert self.emb_dim['entity'] == self.reshape[0] * self.reshape[1]
        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim['entity'])
        self.R = torch.nn.Embedding(self.relation_cnt*2, self.emb_dim['relation'])
        self.input_drop = torch.nn.Dropout(kwargs.get('input_dropout'))
        self.feature_map_drop = torch.nn.Dropout2d(kwargs.get('feature_map_dropout'))
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.bn0 = torch.nn.BatchNorm2d(1)  # batch normalization over a 4D input
        self.bn1 = torch.nn.BatchNorm2d(self.conv_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim['entity'])
        self.bn3 = torch.nn.BatchNorm1d(self.emb_dim['relation'])
        self.register_parameter('b', Parameter(torch.zeros(self.entity_cnt)))
        self.register_parameter('c', Parameter(torch.ones(1)))
        self.filtered = [(self.reshape[0] - self.kernel_size[0]) // self.stride + 1,
                         (self.reshape[1] - self.kernel_size[1]) // self.stride + 1]
        fc_length = self.conv_out_channels * self.filtered[0] * self.filtered[1]
        self.fc = torch.nn.Linear(fc_length, self.emb_dim['entity'])
        self.loss = ConvRLoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)
        self.init()
    
    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector/scalar
        return unitized_vector
    
    def forward(self, batch_h, batch_r, batch_t=None):
        batch_size = batch_h.size(0)
        
        e1 = self.unitized(self.E(batch_h)).view(-1, 1, *self.reshape)
        e1 = self.bn0(e1).view(1, -1, *self.reshape)
        e1 = self.input_drop(e1)

        r = self.R(batch_r)
        r = self.bn3(r)
        r = self.input_drop(r)
        r = r.view(-1, 1, *self.kernel_size)

        x = F.conv2d(e1, r, groups=batch_size)
        x = x.view(batch_size, self.conv_out_channels, *self.filtered)
        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = F.leaky_relu(x, 0.1)

        # x = x.sum(dim=1)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.bn2(x)
        x = self.hidden_drop(x)
        x = self.unitized(x)*self.c
        # x = F.relu(x)
        entities_embedding = self.unitized(self.E.weight)
        x = torch.mm(x, entities_embedding.transpose(1, 0))
        # x += self.b.expand_as(x)
        y = torch.sigmoid(x)
        return self.loss(y, batch_t), y

class ConvRLoss(BaseModel):
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
            loss =  self.loss(batch_p, batch_e) / batch_size
        return loss