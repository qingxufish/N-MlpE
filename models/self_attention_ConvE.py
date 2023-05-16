import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel
from .attention_packge import SelfAttention


class self_attention_ConvE(BaseModel):
    def __init__(self, config):
        super(self_attention_ConvE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')
        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim)
        self.R = torch.nn.Embedding(self.relation_cnt * 2, self.emb_dim)
        self.input_drop = torch.nn.Dropout(kwargs.get('input_dropout'))
        self.feature_map_drop = torch.nn.Dropout2d(kwargs.get('feature_map_dropout'))
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.split_length = kwargs.get('split_length')
        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.reshape = kwargs.get('reshape')
        self.kernel_size = kwargs.get('conv_kernel_size')

        # self.encoder1 = torch.nn.TransformerEncoderLayer(self.split_length, 4, self.split_length, 0.0)
        self.encoder1 = SelfAttention(1, self.split_length[0], self.split_length[0], 0.0)
        self.encoder2 = SelfAttention(1, self.split_length[1], self.split_length[1], 0.0)
        self.encoder3 = SelfAttention(1, self.split_length[2], self.split_length[2], 0.0)

        self.stride = kwargs.get('stride')
        self.Unfold1 = torch.nn.Unfold(kernel_size=(1, self.split_length[0]), stride=(1, self.stride[0]))
        self.Unfold2 = torch.nn.Unfold(kernel_size=(1, self.split_length[1]), stride=(1, self.stride[1]))
        self.Unfold3 = torch.nn.Unfold(kernel_size=(1, self.split_length[2]), stride=(1, self.stride[2]))

        self.bn0 = torch.nn.BatchNorm2d(1)  # batch normalization over a 4D input
        self.bn1 = torch.nn.BatchNorm2d(self.conv_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)
        self.bn3 = torch.nn.BatchNorm2d(1)
        self.register_parameter('b', Parameter(torch.zeros(self.entity_cnt)))
        self.register_parameter('c', Parameter(torch.ones(1)))

        #filtered_h = (self.reshape[0] - self.kernel_size[0]) // 1 + 1
        #filtered_w = (self.reshape[1] - self.kernel_size[1]) // 1 + 1
        #fc_length = self.conv_out_channels * filtered_h * filtered_w
        self.patch_num1 = (self.emb_dim * 2 - self.split_length[0]) // self.stride[0] + 1
        fc_length1 = self.patch_num1*self.split_length[0]

        self.patch_num2 = (self.emb_dim * 2 - self.split_length[1]) // self.stride[1] + 1
        fc_length2 = self.patch_num2*self.split_length[1]

        self.patch_num3 = (self.emb_dim * 2 - self.split_length[2]) // self.stride[2] + 1
        fc_length3 = self.patch_num3*self.split_length[2]

        self.fc = torch.nn.Linear(fc_length1+fc_length2+fc_length3, self.emb_dim)
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
        stacked_inputs = torch.cat([e1, r], 1).reshape(batch_size, 1, 1, -1)  # (batch_size, 1, 2*emb_dim1, emb_dim2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.input_drop(stacked_inputs)

        x1 = self.Unfold1(x).permute(0, 2, 1)
        x2 = self.Unfold2(x).permute(0, 2, 1)
        x3 = self.Unfold3(x).permute(0, 2, 1)

        x1 = self.encoder1(x1).contiguous().view(batch_size, -1)
        x2 = self.encoder2(x2).contiguous().view(batch_size, -1)
        x3 = self.encoder3(x3).contiguous().view(batch_size, -1)

        x = torch.cat((x1, x2, x3), dim=1)
        #x1 = x1.view(batch_size, -1)
        #x1 = x1.view(batch_size, 1, *self.reshape)
        #x = self.bn3(x1+x0)
        #x = self.conv1(x)  # (batch_size, conv_out_channels, 2*emb_dim1-2, emb_dim2-2)
        #x = self.bn1(x)
        # x = x.view(batch_size, -1)  # (batch_size, hidden_size)
        x = self.fc(x)  # (batch_size, embedding_dim)
        x = self.bn2(x)
        x = self.hidden_drop(x)
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