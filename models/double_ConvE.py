import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel
import pickle
import copy


class logic_attention_layer(BaseModel):  # model logic rule by attention which is calculated between relations
    def __init__(self, config):
        super(logic_attention_layer, self).__init__(config)
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
        x = torch.zeros([n_r.shape[0], self.attention_w_out_dim*2], device=self.device)
        x[:, :self.attention_w_out_dim] = torch.mm(n_r, self.w)
        x[:, self.attention_w_out_dim:] = torch.zeros([n_r.shape[0], self.attention_w_out_dim], device=self.device) + \
                                                         torch.mm(goal_r.view(-1, self.emb_dim), self.w)
        x = self.leakyrelu(x)
        x = self.fc(x)
        x = self.softmax(x)
        # x = torch.sigmoid(x)

        return x


class entity_attention_layer(BaseModel):  # model logic rule by attention which is calculated between relations
    def __init__(self, config):
        super(entity_attention_layer, self).__init__(config)
        self.device = config.get('device')
        kwargs = config.get('model_hyper_params')
        self.attention_w_out_dim = kwargs.get('attention_w_out_dim')
        self.emb_dim = kwargs.get('emb_dim')
        self.w = torch.nn.Parameter(torch.Tensor(self.emb_dim, self.attention_w_out_dim))
        self.fc = torch.nn.Linear(self.attention_w_out_dim*2, 1)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        # self.softmax = torch.nn.Softmax(dim=0)
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, n_e, goal_e):
        x = torch.zeros([n_e.shape[0], self.attention_w_out_dim*2], device=self.device)
        x[:, :self.attention_w_out_dim] = torch.mm(n_e, self.w)
        x[:, self.attention_w_out_dim:] = torch.zeros([n_e.shape[0], self.attention_w_out_dim], device=self.device) + \
                                                         torch.mm(goal_e.view(-1, self.emb_dim), self.w)
        x = self.leakyrelu(x)
        x = self.fc(x)
        # x = self.softmax(x)
        x = torch.sigmoid(x)

        return x


class entity_Conv_layer(BaseModel):  # model logic rule by attention which is calculated between relations
    def __init__(self, config):
        super(entity_Conv_layer, self).__init__(config)
        self.device = config.get('device')
        kwargs = config.get('model_hyper_params')
        self.attention_w_out_dim = kwargs.get('attention_w_out_dim')
        self.emb_dim = kwargs.get('emb_dim')

        self.w = torch.nn.Parameter(torch.Tensor(self.emb_dim, self.attention_w_out_dim))
        self.conv = torch.nn.Conv2d()
        self.fc = torch.nn.Linear(self.attention_w_out_dim*2, 1)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        # self.softmax = torch.nn.Softmax(dim=0)
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, n_e, goal_e):
        x = torch.zeros([n_e.shape[0], self.attention_w_out_dim*2], device=self.device)
        x[:, :self.attention_w_out_dim] = torch.mm(n_e, self.w)
        x[:, self.attention_w_out_dim:] = torch.zeros([n_e.shape[0], self.attention_w_out_dim], device=self.device) + \
                                                         torch.mm(goal_e.view(-1, self.emb_dim), self.w)
        x = self.leakyrelu(x)
        x = self.fc(x)
        # x = self.softmax(x)
        x = torch.sigmoid(x)

        return x

class double_ConvE(BaseModel):
    def __init__(self, config):
        super(double_ConvE, self).__init__(config)
        self.device = config.get('device')
        self.single_predict_model = single_ConvE_infer(config)
        self.multi_infer_model = multi_ConvE_infer(config, self.single_predict_model)

    def forward(self, batch_n=None, batch_n_r=None, batch_h=None, batch_r=None, batch_t=None, batch_flag=None, batch_multi_infer=None):
        # single_predict_model train: batch_n, batch_n_r,batch_h, batch_r, batch_t,
        #                             batch_flag, batch_multi_infer=None
        if batch_t is not None and batch_multi_infer is None:
            loss, y = self.single_predict_model(batch_n, batch_n_r, batch_h, batch_r, batch_t, batch_flag)
            return loss, y
        # multi_infer_model train: batch_n=None, batch_n_r=None, batch_h=None, batch_r=None,
        #                          batch_t=None, batch_flag=None, batch_multi_infer
        elif batch_multi_infer is not None:
            loss, y = self.multi_infer_model(batch_multi_infer=batch_multi_infer)
            return loss, y
        # multi_infer_model infer: batch_n=None, batch_n_r=None, batch_h, batch_r,
        #                          batch_t=None, batch_flag=None ,batch_multi_infer=None
        else:
            loss, y = self.multi_infer_model(batch_h=batch_h, batch_r=batch_r)
            return loss, y


class single_ConvE_infer(BaseModel):
    def __init__(self, config):
        super(single_ConvE_infer, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')

        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim)
        self.R = torch.nn.Embedding(self.relation_cnt * 2 + 1, self.emb_dim)  # 反向为另一种关系
        self.init()

        self.entity_attention = entity_attention_layer(config)
        self.entity_attention_loss = torch.nn.MSELoss()

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
        self.b = torch.nn.Parameter(torch.Tensor(self.entity_cnt))
        filtered_h = (self.reshape[0] * 3 - self.kernel_size[0]) // self.stride + 1
        filtered_w = (self.reshape[1] - self.kernel_size[1]) // self.stride + 1
        fc_length = self.conv_out_channels * filtered_h * filtered_w
        self.fc = torch.nn.Linear(fc_length, self.emb_dim)

        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    def entity_loss(self, batch_n, batch_h):
        n_tensor = self.E(batch_n)
        h_tensor = self.E(batch_h)
        # random negative sample
        n_rand_tensor = self.E(torch.randint(0, self.entity_cnt, size=batch_n.shape, device=self.device))
        h_rand_tensor = self.E(torch.randint(0, self.entity_cnt, size=batch_h.shape, device=self.device))
        infer_attention = self.entity_attention(n_tensor, h_tensor)
        rand_infer_attention = self.entity_attention(n_rand_tensor, h_rand_tensor)
        attention = torch.cat((infer_attention, rand_infer_attention), dim=0)
        # calculate loss
        positive_attention = torch.ones_like(infer_attention, device=self.device)
        negative_attention = torch.zeros_like(rand_infer_attention, device=self.device)
        goal_attention = torch.cat((positive_attention, negative_attention), dim=0)

        loss = self.entity_attention_loss(attention, goal_attention)
        return loss

    def forward(self, batch_n=None, batch_n_r=None, batch_h=None, batch_r=None, batch_t=None, batch_flag=None):
        if batch_t is None:
            with torch.no_grad():  # predict
                batch_size = batch_h.size(0)
                m1 = self.E(batch_n).view(-1, 1, *self.reshape)
                e1 = self.E(batch_h).view(-1, 1, *self.reshape)
                r = self.R(batch_r).view(-1, 1, *self.reshape)

                stacked_inputs = torch.cat([m1, e1, r], 2)  # (batch_size, 1, 2*emb_dim1, emb_dim2)

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
                infer_loss = self.entity_loss(batch_n, batch_h)
        else:  # train
            batch_size = batch_h.size(0)
            m1 = self.E(batch_n).view(-1, 1, *self.reshape)
            e1 = self.E(batch_h).view(-1, 1, *self.reshape)
            r = self.R(batch_r).view(-1, 1, *self.reshape)

            stacked_inputs = torch.cat([m1, e1, r], 2)  # (batch_size, 1, 2*emb_dim1, emb_dim2)

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
            infer_loss = self.entity_loss(batch_n, batch_h)
        return self.loss(batch_p=y, infer_loss=infer_loss, batch_t=batch_t), y


class multi_ConvE_infer(BaseModel):
    def __init__(self, config, single_infer_model):
        super(multi_ConvE_infer, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')

        self.logic_attention = logic_attention_layer(config)
        self.single_infer_model = single_infer_model

        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)

        f = open(kwargs.get('goal_data_path'), 'rb')
        self.attention_loss = torch.nn.MSELoss()
        self.neighbor_data = pickle.load(f)
        f.close()

    def p_to_rankp(self, y_list):
        _, index = torch.sort(y_list, 1, descending=True)
        return index

    def batch_multi_infer_model(self, batch_multi_infer):
        total_y = torch.zeros([len(batch_multi_infer), self.entity_cnt], device=self.device)
        batch_t = torch.zeros([len(batch_multi_infer)], dtype=torch.long, device=self.device)
        for infer_ind, infer_triplets_list in enumerate(batch_multi_infer):
            if len(infer_triplets_list) < 2:
                infer_triplets_list.append(infer_triplets_list[0])
            infer_triplets_tensor = torch.tensor(infer_triplets_list).to(self.device)
            n = infer_triplets_tensor[:, 0]
            n_r = infer_triplets_tensor[:, 1]
            h = infer_triplets_tensor[:, 2]
            r = infer_triplets_tensor[:, 3]
            t = infer_triplets_tensor[:, 4]

            _, predict_y = self.single_infer_model(batch_n=n, batch_h=h, batch_r=r)
            predict_attention = self.logic_attention(self.single_infer_model.R(n_r), self.single_infer_model.R(r))
            y = torch.mm(predict_attention.T, predict_y)
            total_y[infer_ind, :] = y
            batch_t[infer_ind] = t[0]
        return total_y, batch_t

    def batch_predict(self, batch_h, batch_r):
        y = torch.zeros([batch_h.shape[0], self.entity_cnt], device=self.device)

        for batch_ind, head_id in enumerate(batch_h):
            rel_id = batch_r[batch_ind]
            _, _, neighbour_id_list, neighbour_rel_id_list = self.find_neighbour_and_rel(head_id)

            entity_attention = self.single_infer_model.entity_attention(self.single_infer_model.E(neighbour_id_list),
                                                                        self.single_infer_model.E(head_id))
            for limit_ind in range(9, -1, -1):
                selected_neighbour_list = entity_attention[:, 0] > 0.1*limit_ind
                selected_neighbour_ind = torch.nonzero(selected_neighbour_list == 1).squeeze()
                if len(selected_neighbour_ind.view(-1)) > 0:
                    break
                else:
                    continue

            neighbour_id_list = neighbour_id_list[selected_neighbour_ind].view(-1)
            neighbour_rel_id_list = neighbour_rel_id_list[selected_neighbour_ind].view(-1)

            if len(neighbour_id_list) < 2:
                neighbour_id_list = torch.cat((neighbour_id_list, neighbour_id_list), dim=0)
                neighbour_rel_id_list = torch.cat((neighbour_rel_id_list, neighbour_rel_id_list), dim=0)

            rel_attention = self.logic_attention(self.single_infer_model.R(neighbour_rel_id_list),
                                                 self.single_infer_model.R(rel_id))

            rel_id_list = torch.zeros_like(neighbour_id_list) + rel_id
            head_id_list = torch.zeros_like(neighbour_id_list) + head_id
            _, y_list = self.single_infer_model(batch_n=neighbour_id_list, batch_h=head_id_list, batch_r=rel_id_list)
            y[batch_ind, :] = torch.mm(rel_attention.T, y_list)

            return y

    def find_neighbour_and_rel(self, e_ind, e_r=None, e_t=None):  # 不能包含训练集的准确实体
        rel_entity_list = copy.deepcopy(self.neighbor_data[e_ind, :])
        neighbour_tensor = torch.tensor([], device=self.device)
        rel_tensor = torch.tensor([], device=self.device)
        neighbour_id_list = torch.tensor([], dtype=torch.int64, device=self.device)
        relation_id_list = torch.tensor([], dtype=torch.int64, device=self.device)

        if e_r is not None:
            e_r = int(e_r)
        for rel_ind, neighbour_list in enumerate(rel_entity_list):
            if neighbour_list is not None:
                if rel_ind != e_r:
                    neighbour_list = torch.tensor(neighbour_list, device=self.device).to(torch.long)
                    rel_ind = torch.tensor(list([rel_ind]), device=self.device).to(torch.long)

                    temp_neighbour_tensor = self.single_infer_model.E(neighbour_list)
                    temp_rel_tensor = torch.zeros_like(temp_neighbour_tensor) + self.single_infer_model.R(rel_ind)

                    neighbour_tensor = torch.cat((neighbour_tensor, temp_neighbour_tensor), dim=0)
                    rel_tensor = torch.cat((rel_tensor, temp_rel_tensor), dim=0)

                    rel_ind_like_neighbour_list = torch.zeros_like(neighbour_list, device=self.device,
                                                                   dtype=torch.int64) + rel_ind

                    neighbour_id_list = torch.cat((neighbour_id_list, neighbour_list), dim=0)
                    relation_id_list = torch.cat((relation_id_list, rel_ind_like_neighbour_list.view(-1)), dim=0)
                else:
                    neighbour_list.remove(float(e_t))  # 去除再训练集内的预测点
                    neighbour_list = torch.tensor(neighbour_list, device=self.device).to(torch.long)
                    rel_ind = torch.tensor(list([rel_ind]), device=self.device).to(torch.long)

                    temp_neighbour_tensor = self.single_infer_model.E(neighbour_list)
                    temp_rel_tensor = torch.zeros_like(temp_neighbour_tensor) + self.single_infer_model.R(rel_ind)

                    neighbour_tensor = torch.cat((neighbour_tensor, temp_neighbour_tensor), dim=0)
                    rel_tensor = torch.cat((rel_tensor, temp_rel_tensor), dim=0)

                    rel_ind_like_neighbour_list = torch.zeros_like(neighbour_list, device=self.device,
                                                                   dtype=torch.int64) + rel_ind

                    neighbour_id_list = torch.cat((neighbour_id_list, neighbour_list), dim=0)
                    relation_id_list = torch.cat((relation_id_list, rel_ind_like_neighbour_list.view(-1)), dim=0)
            else:
                continue

        neighbour_list = e_ind.view(-1, 1)
        rel_ind = torch.tensor(list([self.relation_cnt * 2]), device=self.device).to(torch.long)

        temp_neighbour_tensor = self.single_infer_model.E(neighbour_list)
        temp_rel_tensor = self.single_infer_model.R(rel_ind)

        neighbour_tensor = torch.cat((neighbour_tensor, temp_neighbour_tensor.view(1, -1)), dim=0)
        rel_tensor = torch.cat((rel_tensor, temp_rel_tensor), dim=0)

        neighbour_id_list = torch.cat((neighbour_id_list, neighbour_list.view(-1)), dim=0)
        relation_id_list = torch.cat((relation_id_list, rel_ind.view(-1)), dim=0)

        return neighbour_tensor, rel_tensor, neighbour_id_list, relation_id_list

    def forward(self, batch_h=None, batch_r=None, batch_multi_infer=None):
        # train
        if batch_multi_infer is not None:
            y, batch_t = self.batch_multi_infer_model(batch_multi_infer)
            return self.loss(batch_p=y, batch_t=batch_t), y
        else:
            y = self.batch_predict(batch_h, batch_r)
            return self.loss(batch_p=y, batch_t=None), y


class ConvELoss(BaseModel):
    def __init__(self, device, label_smoothing, entity_cnt):
        super().__init__()
        self.device = device
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.label_smoothing = label_smoothing
        self.entity_cnt = entity_cnt

    def forward(self, batch_p, infer_loss=None, batch_t=None):
        batch_size = batch_p.shape[0]
        loss = None
        if batch_t is not None:
            batch_e = torch.zeros(batch_size, self.entity_cnt, device=self.device).scatter_(1, batch_t.view(-1, 1), 1)
            batch_e = (1.0 - self.label_smoothing) * batch_e + self.label_smoothing / self.entity_cnt
            loss = self.loss(batch_p, batch_e) / batch_size
            if infer_loss is not None:
                return loss + infer_loss
            else:
                return loss

