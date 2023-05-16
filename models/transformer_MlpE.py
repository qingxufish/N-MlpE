import copy
import pickle
import torch
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel


class transformer_MlpE(BaseModel):
    def __init__(self, config):
        super(transformer_MlpE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.ent_dim = kwargs.get('entity_dim')
        self.rel_dim = kwargs.get('relation_dim')

        self.E = torch.nn.Embedding(self.entity_cnt, self.ent_dim)
        self.extractToken = torch.nn.Embedding(1, self.ent_dim*4) # 特殊token加在transformer输入的最后
        self.R = torch.nn.Embedding(self.relation_cnt*2+1, self.rel_dim)  # 反向为另一种关系
        self.init()

        path_kwargs = config.get('pathEmbParams')
        self.nhead = path_kwargs.get('head')
        self.layer_num = path_kwargs.get('layer_num')
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.ent_dim*4, nhead=self.nhead)
        self.pathEmbLayer = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.layer_num)

        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn2 = torch.nn.BatchNorm1d(1)
        self.register_parameter('c', Parameter(torch.ones(1)))

        message_model_kwarg = config.get('MlpE_params')
        self.feature_num = message_model_kwarg.get('feature_num')

        self.encoder = torch.nn.Linear((self.ent_dim + self.rel_dim)*2, self.feature_num)
        self.activate = torch.nn.LeakyReLU(0.1)
        self.decoder = torch.nn.Linear(self.feature_num, self.ent_dim)

        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)

        f = open(kwargs.get('goal_data_path'), 'rb')
        self.neighbor_data = pickle.load(f)
        f.close()

        self.infer_data = {}

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector / scalar
        return unitized_vector

    def find_neighbour_and_rel(self, e_ind, e_r=None, e_t=None):  # 不能包含训练集的准确实体
        rel_entity_list = copy.deepcopy(self.neighbor_data[e_ind, :])
        device = e_ind.device
        neighbour_tensor = torch.tensor([]).to(device)
        rel_tensor = torch.tensor([]).to(device)

        if e_r is not None:
            e_r = int(e_r)
        for rel_ind, neighbour_list in enumerate(rel_entity_list):
            if neighbour_list is not None:
                if rel_ind != e_r:
                    neighbour_list = torch.tensor(neighbour_list).to(torch.long).to(device)
                    rel_ind = torch.tensor(list([rel_ind])).to(torch.long).to(device)

                    temp_neighbour_tensor = self.E(neighbour_list)
                    temp_rel_tensor = torch.zeros_like(temp_neighbour_tensor) + self.R(rel_ind)

                    neighbour_tensor = torch.cat((neighbour_tensor, temp_neighbour_tensor), dim=0)
                    rel_tensor = torch.cat((rel_tensor, temp_rel_tensor), dim=0)
                else:
                    neighbour_list.remove(float(e_t))  # 去除再训练集内的预测点
                    neighbour_list = torch.tensor(neighbour_list).to(torch.long).to(device)
                    rel_ind = torch.tensor(list([rel_ind])).to(torch.long).to(device)

                    temp_neighbour_tensor = self.E(neighbour_list)
                    temp_rel_tensor = torch.zeros_like(temp_neighbour_tensor) + self.R(rel_ind)

                    neighbour_tensor = torch.cat((neighbour_tensor, temp_neighbour_tensor), dim=0)
                    rel_tensor = torch.cat((rel_tensor, temp_rel_tensor), dim=0)
            else:
                continue

        return neighbour_tensor, rel_tensor

    def aggregate_message_by_transformer(self, n_n, n_n_r, n_h, n_r, MhE_interval):
        all_aggregated_message = torch.tensor([]).to(self.device)
        e_ind = 0
        for ind, interval in enumerate(MhE_interval):
            s_ind = e_ind
            e_ind = s_ind + int(interval)
            g_n_n = n_n[s_ind:e_ind, :]
            g_n_n_r = n_n_r[s_ind:e_ind, :]
            g_n_h = n_h[s_ind:e_ind, :]
            g_n_r = n_r[s_ind:e_ind, :]
            all_aggregated_message = torch.cat([all_aggregated_message, self.pathEmbLayer(torch.cat([torch.cat([g_n_n,g_n_n_r,g_n_h,g_n_r],dim=1),
                                              self.extractToken(torch.tensor(0).to(self.device)).view(1,-1)],
                                             dim=0))[-1,:].view(1,-1)], dim=0)

        return all_aggregated_message

    def batch_find_neighbour_and_rel(self,  batch_h, batch_r, batch_t):
        try: #TODO:尝试存储索引信息，加速训练过程
            nn = batch_h[1]  # 假代码，保证Transformer_MlpE.py能够被引入
        except KeyError:
            MhE_n_n = torch.tensor([]).to(self.device)
            MhE_n_n_r = torch.tensor([]).to(self.device)
            MhE_n_h = torch.tensor([]).to(self.device)
            MhE_n_r = torch.tensor([]).to(self.device)
            MhE_interval = torch.tensor([]).to(self.device)
            for batch_ind, head_ind in enumerate(batch_h):
                if batch_t is not None:  # 处于训练阶段，则排除训练集的尾实体
                    rel_ind = batch_r[batch_ind]
                    tail_ind = batch_t[batch_ind]
                    neighbour_tensor, rel_tensor = self.find_neighbour_and_rel(head_ind, rel_ind, tail_ind)
                else:  # 预测阶段，聚合所有邻居的信息
                    rel_ind = batch_r[batch_ind]
                    neighbour_tensor, rel_tensor = self.find_neighbour_and_rel(head_ind)

                if neighbour_tensor.shape[0] > 0:
                    # sigma(alpha*en)
                    n_n = neighbour_tensor
                    n_n_r = rel_tensor
                    n_h = self.E(head_ind).expand_as(n_n).detach()
                    n_r = self.R(rel_ind).expand_as(n_n).detach()
                else:  # if there is only one neighbour, take source node as neighbour and relation is the last one
                    # 0
                    neighbour_tensor = self.E(head_ind)
                    rel_tensor = self.R(torch.tensor(self.relation_cnt*2, device=self.device)).detach().view(1,-1)
                    n_n = neighbour_tensor.view(1,-1)
                    n_n_r = rel_tensor.expand_as(n_n)
                    n_h = self.E(head_ind).expand_as(n_n).detach()
                    n_r = self.R(rel_ind).expand_as(n_n).detach()


                MhE_n_n = torch.cat((MhE_n_n, n_n), dim=0)
                MhE_n_n_r = torch.cat((MhE_n_n_r, n_n_r), dim=0)
                MhE_n_h = torch.cat((MhE_n_h, n_h), dim=0)
                MhE_n_r = torch.cat((MhE_n_r, n_r), dim=0)
                MhE_interval = torch.cat((MhE_interval, torch.tensor(n_n.shape[0], device=self.device).view(1,-1)), dim=0)

            # 使用transformer计算出子图的特征
            e1 = self.aggregate_message_by_transformer(n_n=MhE_n_n,
                                                 n_n_r=MhE_n_n_r,
                                                 n_h=MhE_n_h,
                                                 n_r=MhE_n_r,
                                                 MhE_interval=MhE_interval)
            return e1

    def forward(self, batch_h, batch_r, batch_t=None):

        x = self.batch_find_neighbour_and_rel(batch_h, batch_r, batch_t)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        x = self.bn0(x)
        # x = self.input_drop(x)
        x = self.encoder(x)
        x = self.activate(x)
        # x = self.feature_drop(x)
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

