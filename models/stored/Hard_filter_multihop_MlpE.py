import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel
import pickle
import copy
from .attention_packge import SA_attention, FC_attention, ALL_attention

class Hard_filter_multihop_MlpE(BaseModel):
    def __init__(self, config):
        super(Hard_filter_multihop_MlpE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.ent_dim = kwargs.get('entity_dim')
        self.rel_dim = kwargs.get('relation_dim')
        self.k = kwargs.get('k')
        self.head_num = kwargs.get('head_num')
        exec("self.multi_hop_message = "+kwargs.get("message_model")+"(config)")

        self.E = torch.nn.Embedding(self.entity_cnt, self.ent_dim)
        self.R = torch.nn.Embedding(self.relation_cnt*2+1, self.rel_dim)  # 反向为另一种关系
        self.init()

        exec("self.attention = "+kwargs.get("attention_model")+"(config)")
        exec("self.inner_attention = " + kwargs.get("inner_attention_model") + "(config)")
        exec("self.pooling = self."+kwargs.get("pooling_model"))
        self.k = kwargs.get("k")

        self.softmax = torch.nn.Softmax(dim=0)

        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))

        self.bn2 = torch.nn.BatchNorm1d(1)
        self.register_parameter('c', Parameter(torch.ones(1)))

        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)

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

    def aggregate_message_by_group(self, message, n_n, n_n_r, n_h, n_r, MhE_interval):
        # 按照attention值进行消息聚合，其中attention的和为1
        if isinstance(self.inner_attention, SA_attention):  # 若注意力机制为自注意力
            all_aggregated_message = torch.tensor([]).to(self.device)
            e_ind = 0
            for ind, interval in enumerate(MhE_interval):
                s_ind = e_ind
                e_ind = s_ind + int(interval)
                group_message = message[s_ind:e_ind, :]
                g_n_n = n_n[s_ind:e_ind, :]
                g_n_n_r = n_n_r[s_ind:e_ind, :]
                g_n_h = n_h[s_ind:e_ind, :]
                g_n_r = n_r[s_ind:e_ind, :]
                attention_value = self.inner_attention(n_n=g_n_n, n_n_r=g_n_n_r, n_h=g_n_h, n_r=g_n_r)

                aggregated_message  = torch.mm(attention_value.T, group_message).view(1,-1)
                all_aggregated_message = torch.cat((all_aggregated_message,
                                                    aggregated_message),
                                                   dim=0)
        else:
            all_aggregated_message = torch.tensor([]).to(self.device)
            all_attention_value = self.inner_attention(n_n, n_n_r, n_h, n_r)
            e_ind = 0
            for ind, interval in enumerate(MhE_interval):
                s_ind = e_ind
                e_ind = s_ind + int(interval)
                group_message = message[s_ind:e_ind, :]
                attention_value, _, _ = self.pooling(all_attention_value[s_ind:e_ind, :])

                aggregated_message = torch.mm(attention_value.T, group_message).view(1, -1)
                all_aggregated_message = torch.cat((all_aggregated_message,
                                                    aggregated_message),
                                                   dim=0)

        return all_aggregated_message

    def topk(self, attention_value):
        k = self.cal_k(attention_value, 'max')

        if attention_value.shape[0] > k:
            topk_value, ind = torch.topk(attention_value, k=int(k), dim=0)
        else:
            topk_value = attention_value
            ind = torch.tensor(list(range(0,attention_value.shape[0])), device=self.device, dtype=torch.long).view(-1, 1)

        cof = self.softmax(topk_value)
        return cof, ind, ind.shape[0]

    def cal_k(self, attention_value, method:str):
        if method is 'var':
            # 根据attention的方差计算需要的k值，最大的k是超参数的5倍
            attention_var = torch.var(attention_value, unbiased=False)
            k = (torch.ceil(self.k / (0.2 + attention_var))).item()
        if method is 'max':
            # 根据attention的最大值计算需要的k值，最大的k是超参数的5倍
            path_confidence = torch.sigmoid(attention_value)
            max_confidence = torch.max(path_confidence)
            cof = max_confidence/(1-max_confidence)
            k = torch.ceil(self.k / (0.2 + cof))
        return k

    def hard_filter(self, n_n, n_n_r, n_h, n_r, MhE_interval):
        all_attention_value = self.attention(n_n, n_n_r, n_h, n_r)
        re_n_n = torch.tensor([]).to(self.device)
        re_n_n_r = torch.tensor([]).to(self.device)
        re_n_h = torch.tensor([]).to(self.device)
        re_n_r = torch.tensor([]).to(self.device)
        re_interval = torch.tensor([]).to(self.device)
        e_ind = 0
        for ind, interval in enumerate(MhE_interval):
            s_ind = e_ind
            e_ind = s_ind + int(interval)
            _, select_ind, select_interval = self.pooling(all_attention_value[s_ind:e_ind, :])

            re_n_n = torch.cat((re_n_n, n_n[s_ind:e_ind, :][select_ind.view(-1), :]), dim=0)
            re_n_n_r = torch.cat((re_n_n_r, n_n_r[s_ind:e_ind, :][select_ind.view(-1), :]), dim=0)
            re_n_h = torch.cat((re_n_h, n_h[s_ind:e_ind, :][select_ind.view(-1), :]), dim=0)
            re_n_r = torch.cat((re_n_r, n_r[s_ind:e_ind, :][select_ind.view(-1), :]), dim=0)
            re_interval = torch.cat((re_interval, torch.tensor(select_interval, device=self.device).view(1,-1)), dim=0)

        return re_n_n, re_n_n_r, re_n_h, re_n_r, re_interval

    def batch_find_neighbour_and_rel(self,  batch_h, batch_r, batch_t):
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
                n_h = self.E(head_ind).expand_as(n_n)
                n_r = self.R(rel_ind).expand_as(n_n)
            else:  # if there is only one neighbour, take source node as neighbour and relation is the last one
                # 0
                neighbour_tensor = self.E(head_ind)
                rel_tensor = self.R(torch.tensor(self.relation_cnt*2, device=self.device)).view(1,-1)
                n_n = neighbour_tensor.view(1,-1)
                n_n_r = rel_tensor.expand_as(n_n)
                n_h = self.E(head_ind).expand_as(n_n)
                n_r = self.R(rel_ind).expand_as(n_n)


            MhE_n_n = torch.cat((MhE_n_n, n_n), dim=0)
            MhE_n_n_r = torch.cat((MhE_n_n_r, n_n_r), dim=0)
            MhE_n_h = torch.cat((MhE_n_h, n_h), dim=0)
            MhE_n_r = torch.cat((MhE_n_r, n_r), dim=0)
            MhE_interval = torch.cat((MhE_interval, torch.tensor(n_n.shape[0], device=self.device).view(1,-1)), dim=0)

        # 利用attention将贡献非常小的路径删除，加快运行速度
        MhE_n_n, MhE_n_n_r, MhE_n_h, MhE_n_r, MhE_interval = self.hard_filter(n_n=MhE_n_n,n_n_r=MhE_n_n_r,
                                                                              n_h=MhE_n_h,n_r=MhE_n_r,
                                                                              MhE_interval=MhE_interval)

        aggregate_message = self.multi_hop_message(n_n=MhE_n_n,
                                                n_n_r=MhE_n_n_r,
                                                n_h=MhE_n_h,
                                                n_r=MhE_n_r,
                                                )
        e1 = self.aggregate_message_by_group(message=aggregate_message,
                                             n_n=MhE_n_n,
                                             n_n_r=MhE_n_n_r,
                                             n_h=MhE_n_h,
                                             n_r=MhE_n_r,
                                             MhE_interval=MhE_interval)
        return e1

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector / scalar
        return unitized_vector

    def forward(self, batch_h, batch_r, batch_t=None):
        m1 = self.batch_find_neighbour_and_rel(batch_h, batch_r, batch_t)
        x = self.bn2(m1.unsqueeze(1))
        x = self.hidden_drop(x).squeeze()
        x = self.unitized(x)* self.c
        entities_embedding = self.unitized(self.E.weight)
        x = torch.mm(x, entities_embedding.transpose(1, 0))  # *self.c  # c is important
        y = torch.sigmoid(x)
        return self.loss(y, batch_t), y


class Multi_hop_MlpE(BaseModel):
    def __init__(self, config):
        super(Multi_hop_MlpE, self).__init__(config)
        self.device = config.get('device')
        kwargs = config.get('model_hyper_params')
        self.entity_dim = kwargs.get('entity_dim')
        self.relation_dim = kwargs.get('relation_dim')
        self.k = kwargs.get('k')

        self.input_drop = torch.nn.Dropout(kwargs.get('input_dropout'))
        self.feature_drop = torch.nn.Dropout(kwargs.get('feature_map_dropout'))
        self.multi_hop_drop = torch.nn.Dropout(kwargs.get('multi_hop_drop'))

        message_model_kwarg = config.get('MlpE_params')
        self.feature_num = message_model_kwarg.get('feature_num')

        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn2 = torch.nn.BatchNorm1d(1)

        self.encoder = torch.nn.Linear((self.entity_dim + self.relation_dim)*2, self.feature_num)
        self.decoder = torch.nn.Linear(self.feature_num, self.entity_dim)

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector / scalar
        return unitized_vector

    def forward(self, n_n, n_n_r, n_h, n_r):  # neighbour, neighbour relation, head entity, predict relation
        batch_size = n_h.size(0)
        e1 = self.unitized(n_n)
        r1 = self.unitized(n_n_r)
        e = self.unitized(n_h)
        r = self.unitized(n_r)
        stacked_inputs = torch.cat([e1, r1, e, r], 1).view(batch_size, 1, -1)  # (batch_size, 1, 2*emb_dim1, emb_dim2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.input_drop(stacked_inputs)
        x = self.encoder(x)
        x = self.feature_drop(x)
        x = torch.relu(x)
        x = self.decoder(x)
        x = self.bn2(x)
        x = self.multi_hop_drop(x)  # 不确定是否起效

        return x.view(batch_size,-1)


class Multi_hop_ConvE(BaseModel):
    def __init__(self, config):
        super(Multi_hop_ConvE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.entity_dim = kwargs.get('entity_dim')
        self.relation_dim = kwargs.get('relation_dim')
        self.input_drop = torch.nn.Dropout(kwargs.get('input_dropout'))
        self.feature_map_drop = torch.nn.Dropout(kwargs.get('feature_map_dropout'))
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))

        model_kwargs = config.get('ConvE_params')
        self.conv_out_channels = model_kwargs.get('conv_out_channels')
        self.reshape = model_kwargs.get('reshape')
        self.kernel_size = model_kwargs.get('conv_kernel_size')
        self.stride = model_kwargs.get('stride')
        # convolution layer, in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=0
        self.conv1 = torch.nn.Conv3d(1, self.conv_out_channels, self.kernel_size, 1, 0, bias=kwargs.get('use_bias'))
        self.bn0 = torch.nn.BatchNorm3d(1)  # batch normalization over a 5D input
        self.bn1 = torch.nn.BatchNorm3d(self.conv_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.entity_dim)

        filtered_d = (4-(self.kernel_size[0]-1)-1) // self.stride + 1
        filtered_h = (self.reshape[1]-(self.kernel_size[1]-1)-1) // self.stride + 1
        filtered_w = (self.reshape[2]-(self.kernel_size[2]-1)-1) // self.stride + 1
        fc_length = self.conv_out_channels * filtered_h * filtered_w * filtered_d
        self.fc = torch.nn.Linear(fc_length, self.entity_dim)

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector/scalar
        return unitized_vector


    def forward(self, n_n, n_n_r, n_h, n_r):  # neighbour, neighbour relation, head entity, predict relation
        n_size = n_r.size(0)
        m1 = n_n.view(-1, 1, *self.reshape)
        r1 = n_n_r.view(-1, 1, *self.reshape)
        e1 = n_h.view(-1, 1, *self.reshape)
        r =  n_r.view(-1, 1, *self.reshape)
        stacked_inputs = torch.cat([m1, r1, e1, r], dim=2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.input_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = F.leaky_relu(x,0.1)
        x = x.view(n_size, -1)
        x = self.fc(x)
        x = self.bn2(x)

        return x


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