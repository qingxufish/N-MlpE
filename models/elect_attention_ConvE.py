import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel
import pickle
import copy


class attention_layer(BaseModel):
    def __init__(self, config):
        super(attention_layer, self).__init__(config)
        self.device = config.get('device')
        kwargs = config.get('model_hyper_params')
        self.attention_w_out_dim = kwargs.get('attention_w_out_dim')
        self.emb_dim = kwargs.get('emb_dim')
        self.w = torch.nn.Parameter(torch.Tensor(self.emb_dim, self.attention_w_out_dim))
        self.fc = torch.nn.Linear(self.attention_w_out_dim * 2, 1)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        self.softmax = torch.nn.Softmax(0)
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, n_r, goal_r):  # nr:[neighbour_count,embedding_size]
        # 用于轮盘赌的概率
        x = torch.zeros([n_r.shape[0], self.attention_w_out_dim*2]).to(self.device)
        x[:, :self.attention_w_out_dim] = torch.mm(n_r, self.w)
        x[:, self.attention_w_out_dim:] = torch.zeros([n_r.shape[0], self.attention_w_out_dim]).to(self.device) + \
                                                      torch.mm(goal_r, self.w)
        x = self.leakyrelu(x)
        x = self.fc(x)
        # x = torch.sigmoid(x)
        x = self.softmax(x)

        return x


class single_infer_ConvE(BaseModel):
    def __init__(self, config):
        super(single_infer_ConvE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')

        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim)
        self.R = torch.nn.Embedding(self.relation_cnt*2+1, self.emb_dim)  # 反向为另一种关系
        self.link_chart = torch.zeros([self.relation_cnt*2+1, self.relation_cnt*2+1])
        self.link_chart_flag = 1
        self.init()

        self.attention = attention_layer(config)
        self.input_drop = torch.nn.Dropout(kwargs.get('input_dropout'))
        self.feature_map_drop = torch.nn.Dropout3d(kwargs.get('feature_map_dropout'))
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.reshape = kwargs.get('reshape')
        self.kernel_size = kwargs.get('conv_kernel_size')
        self.stride = kwargs.get('stride')
        # convolution layer, in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=0
        self.conv1 = torch.nn.Conv3d(1, self.conv_out_channels, self.kernel_size, 1, 0, bias=kwargs.get('use_bias'))
        self.bn0 = torch.nn.BatchNorm3d(1)  # batch normalization over a 5D input
        self.bn1 = torch.nn.BatchNorm3d(self.conv_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)
        self.b = torch.nn.Parameter(torch.Tensor(self.entity_cnt))
        filtered_d = (4-(self.kernel_size[0]-1)-1) // self.stride + 1
        filtered_h = (self.reshape[1]-(self.kernel_size[1]-1)-1) // self.stride + 1
        filtered_w = (self.reshape[2]-(self.kernel_size[2]-1)-1) // self.stride + 1
        fc_length = self.conv_out_channels * filtered_h * filtered_w * filtered_d
        self.fc = torch.nn.Linear(fc_length, self.emb_dim)
        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)

        f = open(kwargs.get('goal_data_path'), 'rb')
        self.neighbor_data = pickle.load(f)
        f.close()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    def memorize_link(self, b_n_r, b_goal_r):  # b_n_r and b_goal_r are in the same shape
        for batch_ind, n_r in enumerate(b_n_r):
            goal_r = b_goal_r[batch_ind]
            self.link_chart[n_r][goal_r] = 1

    def link_exist_exam(self, b_n_r, b_goal_r):  # b_n_r and b_goal_r are in the same shape
        exist_list = torch.zeros_like(b_n_r)
        for batch_ind, n_r in enumerate(b_n_r):
            goal_r = b_goal_r[batch_ind]
            exist_list[batch_ind] = self.link_chart[n_r][goal_r]
        return exist_list

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

                    temp_neighbour_tensor = self.E(neighbour_list)
                    temp_rel_tensor = torch.zeros_like(temp_neighbour_tensor) + self.R(rel_ind)

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

                    temp_neighbour_tensor = self.E(neighbour_list)
                    temp_rel_tensor = torch.zeros_like(temp_neighbour_tensor) + self.R(rel_ind)

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

        temp_neighbour_tensor = self.E(neighbour_list)
        temp_rel_tensor = self.R(rel_ind)

        neighbour_tensor = torch.cat((neighbour_tensor, temp_neighbour_tensor.view(1, -1)), dim=0)
        rel_tensor = torch.cat((rel_tensor, temp_rel_tensor), dim=0)

        neighbour_id_list = torch.cat((neighbour_id_list, neighbour_list.view(-1)), dim=0)
        relation_id_list = torch.cat((relation_id_list, rel_ind.view(-1)), dim=0)

        return neighbour_tensor, rel_tensor, neighbour_id_list, relation_id_list

    def forward(self, batch_n=None, batch_n_r=None, batch_h=None, batch_r=None, batch_t=None, batch_flag=None):
        batch_size = batch_r.size(0)
        # if batch_t is not None and self.link_chart_flag:
            # self.memorize_link(batch_n_r, batch_r)

        m1 = self.E(batch_n).view(-1, 1, *self.reshape)
        r1 = self.R(batch_n_r).view(-1, 1, *self.reshape)
        e1 = self.E(batch_h).view(-1, 1, *self.reshape)
        r = self.R(batch_r).view(-1, 1, *self.reshape)
        stacked_inputs = torch.cat([m1, r1, e1, r], dim=2)  # (batch_size, 1, 4, emb_dim1, emb_dim2)

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
        return y


class elect_attention_ConvE(BaseModel):
    def __init__(self, config):
        super(elect_attention_ConvE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')

        self.single_predict_model = single_infer_ConvE(config)
        self.attention = attention_layer(config)
        self.softmax = torch.nn.Softmax(0)

        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)

    def dirac_loss_list(self, predict_x, goal_ind):
        loss_tensor = torch.zeros_like(goal_ind)
        for x_ind, x in enumerate(predict_x):
            loss_tensor[x_ind] = self.loss(x.view(1, -1), goal_ind[x_ind])

        return loss_tensor

    def attention_based_infer(self, n, n_r, h, r, t=None):  # 选择方差最大的，也就是找预测最准确的
        predict_x = self.single_predict_model(batch_n=n, batch_n_r=n_r, batch_h=h, batch_r=r, batch_t=t)
        _, goal_ind = torch.max(predict_x, dim=1)
        loss = self.dirac_loss_list(predict_x, goal_ind)
        _, ind = torch.min(loss, dim=0)
        y = predict_x[ind, :]
        return y

    def batch_predict(self, batch_h, batch_r):  # multi predict model
        batch_y = torch.zeros([batch_h.shape[0], self.entity_cnt], device=self.device)
        for batch_ind, head_id in enumerate(batch_h):
            rel_id = batch_r[batch_ind]

            _, _, neighbour_id_list, neighbour_rel_id_list = self.single_predict_model.find_neighbour_and_rel(head_id)

            batch_n = neighbour_id_list
            batch_n_r = neighbour_rel_id_list
            batch_h_ = head_id.expand_as(batch_n)
            batch_r_ = rel_id.expand_as(batch_n)

            batch_y[batch_ind, :] = self.attention_based_infer(batch_n, batch_n_r, batch_h_, batch_r_)
        return batch_y

    def batch_multi_infer_model(self, batch_multi_infer):  # multi predict train
        total_y = torch.zeros([len(batch_multi_infer), self.entity_cnt], device=self.device)
        batch_t = torch.zeros([len(batch_multi_infer)], dtype=torch.long, device=self.device)
        for infer_ind, infer_triplets_list in enumerate(batch_multi_infer):
            infer_triplets_tensor = torch.tensor(infer_triplets_list).to(self.device)
            n = infer_triplets_tensor[:, 0]
            n_r = infer_triplets_tensor[:, 1]
            h = infer_triplets_tensor[:, 2]
            r = infer_triplets_tensor[:, 3]
            t = infer_triplets_tensor[:, 4]

            total_y[infer_ind, :] = self.attention_based_infer(n, n_r, h, r, t)
            batch_t[infer_ind] = t[0]
        return total_y, batch_t

    def forward(self, batch_n=None, batch_n_r=None, batch_h=None, batch_r=None, batch_t=None, batch_flag=None, batch_multi_infer=None):
        if batch_t is not None and batch_h is not None:
            y = self.single_predict_model(batch_n=batch_n, batch_n_r=batch_n_r, batch_h=batch_h, batch_r=batch_r, batch_t=batch_r)
            return self.loss(y, batch_t), y
        elif batch_multi_infer is not None:  # train
            y, batch_t = self.batch_multi_infer_model(batch_multi_infer)
            return self.loss(y, batch_t), y
        else:  # predict
            y = self.batch_predict(batch_h, batch_r)
            return None, y



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