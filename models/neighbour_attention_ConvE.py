import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel
import pickle
import copy


class attention_layer(BaseModel): #  利用邻居节点和预测节点的embedding来计算注意力
    def __init__(self, config):
        super(attention_layer, self).__init__(config)
        self.device = config.get('device')
        kwargs = config.get('model_hyper_params')
        self.attention_w_out_dim = kwargs.get('attention_w_out_dim')
        self.emb_dim = kwargs.get('emb_dim')
        self.w = torch.nn.Parameter(torch.Tensor(self.emb_dim, self.attention_w_out_dim))
        self.fc = torch.nn.Linear(self.attention_w_out_dim*2, 1)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        self.softmax = torch.nn.Softmax(dim=0)
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, n_r, goal_r):  # nr:[neighbour_count,embedding_size]
        # concat_nr_goal_r:[neighbour_count,attention_w_out_dim*2]
        x = torch.zeros([n_r.shape[0], self.attention_w_out_dim*2]).to(self.device)
        x[:, :self.attention_w_out_dim] = torch.mm(n_r, self.w)
        x[:, self.attention_w_out_dim:] = torch.zeros([n_r.shape[0], self.attention_w_out_dim]).to(self.device) + \
                                                         torch.mm(goal_r.view(1, -1), self.w)
        x = self.leakyrelu(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x


class neighbour_attention_ConvE(BaseModel):
    def __init__(self, config):
        super(neighbour_attention_ConvE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')

        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim)
        self.R = torch.nn.Embedding(self.relation_cnt*2+1, self.emb_dim)  # 反向为另一种关系
        self.init()

        self.attention = attention_layer(config)
        self.input_drop = torch.nn.Dropout3d(kwargs.get('input_dropout'))
        self.feature_map_drop = torch.nn.Dropout3d(kwargs.get('feature_map_dropout'))
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.reshape = kwargs.get('reshape')
        self.kernel_size = kwargs.get('conv_kernel_size')
        self.stride = kwargs.get('stride')
        # convolution layer, in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=0
        self.bn0_1 = torch.nn.BatchNorm1d(1)  # batch normalization over a 5D input
        self.bn0_2 = torch.nn.BatchNorm1d(1)
        self.bn0_3 = torch.nn.BatchNorm1d(1)

        self.bn2 = torch.nn.BatchNorm1d(1)
        self.register_parameter('c', Parameter(torch.ones(1)))


        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)

        self.MlpE = torch.nn.Sequential(torch.nn.Linear(self.emb_dim+self.emb_dim, self.emb_dim*self.emb_dim),
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(self.emb_dim*self.emb_dim, self.emb_dim)
                                        )

        self.operation1 = torch.nn.Sequential(torch.nn.Linear(self.emb_dim+self.emb_dim, self.emb_dim*self.emb_dim),
                                        torch.nn.LeakyReLU(0.1)
                                        )
        self.operation2 = torch.nn.Sequential(torch.nn.Linear(self.emb_dim+self.emb_dim, self.emb_dim*self.emb_dim),
                                        torch.nn.LeakyReLU(0.1)
                                        )
        self.operation3 = torch.nn.Sequential(torch.nn.Linear(self.emb_dim+self.emb_dim, self.emb_dim*self.emb_dim),
                                        torch.nn.LeakyReLU(0.1)
                                        )
        self.decoder = torch.nn.Sequential(torch.nn.Linear(self.emb_dim*self.emb_dim*3, self.emb_dim)
                                           )


        f = open(kwargs.get('goal_data_path'), 'rb')
        self.neighbor_data = pickle.load(f)
        f.close()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

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

    def batch_find_neighbour_and_rel(self,  batch_h, batch_r, batch_t):
        e1 = torch.tensor([]).to(self.device)
        for batch_ind, head_ind in enumerate(batch_h):
            if batch_t is not None:  # 处于训练阶段，则排除训练集的尾实体
                rel_ind = batch_r[batch_ind]
                tail_ind = batch_t[batch_ind]
                neighbour_tensor, rel_tensor = self.find_neighbour_and_rel(head_ind, rel_ind, tail_ind)
            else:  # 预测阶段，聚合所有邻居的信息
                rel_ind = batch_r[batch_ind]
                neighbour_tensor, rel_tensor = self.find_neighbour_and_rel(head_ind)

            if neighbour_tensor.shape[0] > 0:  # 训练阶段，如果存在1个以上的邻居，那么直接将目标信息作为邻居信息
                # sigma(alpha*en)
                attention_tensor = self.attention(rel_tensor, self.R(rel_ind))
                stack_neighbour_rel = torch.cat([neighbour_tensor, rel_tensor], dim=1)
                neighbour_message = self.MlpE(stack_neighbour_rel)
                aggregate_message = torch.mm(attention_tensor.T, neighbour_message)
            else:  # if there is only one neighbour, take source node as neighbour and relation is the last one
                # 0
                neighbour_tensor = self.E(head_ind).view(1,-1)
                rel_tensor = self.R(torch.tensor(self.relation_cnt*2, device=self.device)).view(1,-1)
                stack_neighbour_rel = torch.cat([neighbour_tensor, rel_tensor], dim=1)
                aggregate_message = self.MlpE(stack_neighbour_rel)

            e1 = torch.cat((e1, aggregate_message), dim=0)

        return e1

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector / scalar
        return unitized_vector

    def forward(self, batch_h, batch_r, batch_t=None):
        batch_size = batch_h.size(0)
        m1 = self.batch_find_neighbour_and_rel(batch_h, batch_r, batch_t)

        m1 = self.unitized(m1)
        e1 = self.unitized(self.E(batch_h))
        r = self.unitized(self.R(batch_r))
        stacked_inputs1 = torch.cat([m1, e1], dim=1).view(batch_size, 1, -1)
        stacked_inputs2 = torch.cat([e1, r], dim=1).view(batch_size, 1, -1)
        stacked_inputs3 = torch.cat([m1, r], dim=1).view(batch_size, 1, -1)

        stacked_inputs1 = self.bn0_1(stacked_inputs1)
        stacked_inputs2 = self.bn0_2(stacked_inputs2)
        stacked_inputs3 = self.bn0_3(stacked_inputs3)

        o1 = self.operation1(stacked_inputs1)
        o2 = self.operation2(stacked_inputs2)
        o3 = self.operation3(stacked_inputs3)

        x = torch.cat([o1,o2,o3],dim=2)
        x = self.decoder(x)  # (batch_size, embedding_dim)
        x = self.bn2(x)
        x = self.hidden_drop(x).view(batch_size, -1)
        x = self.unitized(x)* self.c

        entities_embedding = self.unitized(self.E.weight)
        x = torch.mm(x, entities_embedding.transpose(1, 0))  # *self.c  # c is important
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