import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel
from utils import calc_goal_distribute
import pickle


class ConvE_for_GCN(BaseModel):
    def __init__(self, config):
        super(ConvE_for_GCN, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')
        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim)
        self.R = torch.nn.Embedding(self.relation_cnt, self.emb_dim)
        self.input_drop = torch.nn.Dropout(kwargs.get('input_dropout'))
        self.feature_map_drop = torch.nn.Dropout2d(kwargs.get('feature_map_dropout'))
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.reshape = kwargs.get('reshape')
        self.kernel_size = kwargs.get('conv_kernel_size')
        self.stride = kwargs.get('stride')
        # convolution layer, in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=0
        self.conv1 = torch.nn.Conv2d(1, self.conv_out_channels, self.kernel_size, 1, 0, bias=kwargs.get('use_bias'))
        self.bn0 = torch.nn.BatchNorm2d(1)  # batch normalization over a 4D input
        self.bn1 = torch.nn.BatchNorm2d(self.conv_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)
        self.register_parameter('b', Parameter(torch.zeros(self.entity_cnt)))
        filtered_h = (self.reshape[0] * 2 - self.kernel_size[0]) // self.stride + 1
        filtered_w = (self.reshape[1] - self.kernel_size[1]) // self.stride + 1
        fc_length = self.conv_out_channels * filtered_h * filtered_w
        self.fc = torch.nn.Linear(fc_length, self.emb_dim)
        self.loss = ConvE_for_GCNLoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt, self.relation_cnt, kwargs.get('goal_data_path'))
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    def forward(self, batch_h, batch_r, batch_t=None):
        batch_size = batch_h.size(0)
        e1 = self.E(batch_h).view(-1, 1, *self.reshape)
        r = self.R(batch_r).view(-1, 1, *self.reshape)
        stacked_inputs = torch.cat([e1, r], 2)  # (batch_size, 1, 2*emb_dim1, emb_dim2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.input_drop(stacked_inputs)
        x = self.conv1(x)  # (batch_size, conv_out_channels, 2*emb_dim1-2, emb_dim2-2)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)  # (batch_size, hidden_size)
        x = self.fc(x)  # (batch_size, embedding_dim)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        y = torch.sigmoid(x)
        return self.loss(y, batch_h, batch_r, batch_t), y


class ConvE_for_GCNLoss(BaseModel):
    def __init__(self, device, label_smoothing, entity_cnt, relation_cnt,goal_data_path):
        super().__init__()
        self.device = device
        self.loss = torch.nn.CosineEmbeddingLoss(margin=0.1, reduction='sum')
        self.label_smoothing = label_smoothing
        self.entity_cnt = entity_cnt
        self.relation_cnt = relation_cnt

        f = open(goal_data_path, 'rb')
        self.goal_data = pickle.load(f)
        f.close()

    def forward(self, batch_p, batch_h, batch_r, batch_t=None):
        batch_size = batch_p.shape[0]
        loss = None
        if batch_t is not None:
            label = torch.ones_like(batch_p[:, 0])
            goal_dis = torch.tensor(calc_goal_distribute(batch_h, batch_r, batch_t, self.goal_data, self.entity_cnt, self.relation_cnt))
            loss = self.loss(batch_p, goal_dis, label) / batch_size
        return loss