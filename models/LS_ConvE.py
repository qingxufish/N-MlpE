import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel
import pickle
import copy
from .attention_packge import attention_layer


class single_infer(BaseModel):
    def __init__(self, config):
        super(single_infer, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')

        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim)
        self.R = torch.nn.Embedding(self.relation_cnt*2+1, self.emb_dim)  # 反向为另一种关系
        self.init()

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
        self.register_parameter('b', Parameter(torch.zeros(self.entity_cnt)))
        self.register_parameter('c', Parameter(torch.zeros(1)))
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

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector/scalar
        return unitized_vector

    def find_neighbour_and_rel(self, e_ind, e_r=None, e_t=None):  # 不能包含训练集的准确实体
        rel_entity_list = copy.deepcopy(self.neighbor_data[e_ind, :])
        neighbour_id_list = torch.tensor([], dtype=torch.int64, device=self.device)
        relation_id_list = torch.tensor([], dtype=torch.int64, device=self.device)

        if e_r is not None:
            e_r = int(e_r)
        for rel_ind, neighbour_list in enumerate(rel_entity_list):
            if neighbour_list is not None:
                if rel_ind != e_r:
                    neighbour_list = torch.tensor(neighbour_list, device=self.device).to(torch.long)
                    rel_ind = torch.tensor(list([rel_ind]), device=self.device).to(torch.long)

                    rel_ind_like_neighbour_list = rel_ind.expand_as(neighbour_list)

                    neighbour_id_list = torch.cat((neighbour_id_list, neighbour_list), dim=0)
                    relation_id_list = torch.cat((relation_id_list, rel_ind_like_neighbour_list.view(-1)), dim=0)
                else:
                    neighbour_list.remove(float(e_t))  # 去除再训练集内的预测点
                    neighbour_list = torch.tensor(neighbour_list, device=self.device).to(torch.long)
                    rel_ind = torch.tensor(list([rel_ind]), device=self.device).to(torch.long)

                    rel_ind_like_neighbour_list = rel_ind.expand_as(neighbour_list)

                    neighbour_id_list = torch.cat((neighbour_id_list, neighbour_list), dim=0)
                    relation_id_list = torch.cat((relation_id_list, rel_ind_like_neighbour_list.view(-1)), dim=0)
            else:
                continue
        # 每个实体都有一个为自身的邻居，且关系为is
        neighbour_list = e_ind.view(-1, 1)
        rel_ind = torch.tensor(list([self.relation_cnt * 2]), device=self.device).to(torch.long)

        neighbour_id_list = torch.cat((neighbour_id_list, neighbour_list.view(-1)), dim=0)
        relation_id_list = torch.cat((relation_id_list, rel_ind.view(-1)), dim=0)

        return None, None, neighbour_id_list, relation_id_list


    def single_infer2entity(self, n_n, n_n_r, n_h, n_r):  # neighbour, neighbour relation, head entity, predict relation
        n_size = n_r.size(0)
        m1 = self.E(n_n).view(-1, 1, *self.reshape)
        r1 = self.R(n_n_r).view(-1, 1, *self.reshape)
        e1 = self.E(n_h).view(-1, 1, *self.reshape)
        r = self.R(n_r).view(-1, 1, *self.reshape)
        stacked_inputs = torch.cat([m1, r1, e1, r], dim=2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.input_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = F.relu(x)
        x = x.view(n_size, -1)
        x = self.fc(x)
        x = self.bn2(x)
        x = self.hidden_drop(x)
        #x = F.relu(x)
        #x = torch.mm(x, self.E.weight.transpose(1, 0))
        #x += self.b.expand_as(x)
        #y = torch.sigmoid(x)
        return x

    def pre_train(self, batch_h=None, batch_r=None, batch_n=None, batch_n_r=None, batch_t=None):
        n_size = batch_n.size(0)
        m1 = self.E(batch_n).view(-1, 1, *self.reshape)
        r1 = self.R(batch_n_r).view(-1, 1, *self.reshape)
        e1 = self.E(batch_h).view(-1, 1, *self.reshape)
        r = self.R(batch_r).view(-1, 1, *self.reshape)
        stacked_inputs = torch.cat([m1, r1, e1, r], dim=2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.input_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = F.relu(x)
        x = x.view(n_size, -1)
        x = self.fc(x)
        x = self.bn2(x)
        x = self.hidden_drop(x)
        # x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        # x += self.b.expand_as(x)
        y = torch.sigmoid(x)
        return self.loss(y, batch_t), y


class multi_infer(BaseModel):
    def __init__(self, config, single_infer_model):
        super(multi_infer, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')

        self.link_chart = torch.zeros([self.relation_cnt*2+1, self.relation_cnt*2+1])
        self.link_chart_flag = 1

        self.confidence_chart = torch.zeros([self.relation_cnt*2+1, self.relation_cnt*2+1])

        self.pre_train_flag = 1
        self.single_infer = single_infer_model
        self.attention = attention_layer(config)
        self.register_parameter('b', Parameter(torch.zeros(self.entity_cnt)))
        self.register_parameter('c', Parameter(torch.zeros(1)))

        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)
        self.attention_loss = torch.nn.BCELoss()
        f = open(kwargs.get('goal_data_path'), 'rb')
        self.neighbor_data = pickle.load(f)
        f.close()

        f = open(kwargs.get('logic_data_path'), 'rb')
        self.logic_data = pickle.load(f)
        self.complete_confidence()
        f.close()

    def complete_confidence(self):
        for confidence_ele in self.logic_data:
            if len(confidence_ele) < 4:  # add 'is' relation into 2 hops infer
                infer_confidence = confidence_ele[-1][0]
                infer_result = confidence_ele[-2]
                evidence1 = confidence_ele[-3]
                # direction 1
                if infer_result[0] == evidence1[2]:
                    self.confidence_chart[evidence1[1]][infer_result[1]] = infer_confidence
                elif infer_result[0] == evidence1[0]:
                    temp_evidence = self.inverse_triplet(evidence1)
                    self.confidence_chart[temp_evidence[1]][infer_result[1]] = infer_confidence
                # direction 2
                infer_result = self.inverse_triplet(infer_result)
                if infer_result[0] == evidence1[2]:
                    self.confidence_chart[evidence1[1]][infer_result[1]] = infer_confidence
                elif infer_result[0] == evidence1[0]:
                    temp_evidence = self.inverse_triplet(evidence1)
                    self.confidence_chart[temp_evidence[1]][infer_result[1]] = infer_confidence

            else: # 3 hops infer
                loop_ind = [[-3, -4], [-4, -3]]
                for ind in range(0, 2):
                    index = loop_ind[ind]
                    infer_confidence = confidence_ele[-1][0]
                    infer_result = confidence_ele[index[0]]
                    evidence1 = confidence_ele[index[1]]
                    # direction 1
                    if infer_result[0] == evidence1[2]:
                        self.confidence_chart[evidence1[1]][infer_result[1]] = infer_confidence
                    elif infer_result[0] == evidence1[0]:
                        temp_evidence = self.inverse_triplet(evidence1)
                        self.confidence_chart[temp_evidence[1]][infer_result[1]] = infer_confidence
                    # direction 2
                    infer_result = self.inverse_triplet(infer_result)
                    if infer_result[0] == evidence1[2]:
                        self.confidence_chart[evidence1[1]][infer_result[1]] = infer_confidence
                    elif infer_result[0] == evidence1[0]:
                        temp_evidence = self.inverse_triplet(evidence1)
                        self.confidence_chart[temp_evidence[1]][infer_result[1]] = infer_confidence

    def confidence_exam(self, b_n_r, b_goal_r):
        confidence_list = torch.zeros_like(b_n_r, dtype=torch.float)
        for batch_ind, n_r in enumerate(b_n_r):
            goal_r = b_goal_r[batch_ind]
            confidence_list[batch_ind] = self.confidence_chart[n_r][goal_r]
        return confidence_list

    def inverse_triplet(self, triplet):
        if triplet[1] < self.relation_cnt:  # if the triplet is in direction 1
            result = triplet[::-1]
            result[1] = result[1] + self.relation_cnt
        elif triplet[1] >= self.relation_cnt:  # if the triplet is in direction 2
            result = triplet[::-1]
            result[1] = result[1] - self.relation_cnt
        return result

    def memorize_link(self, b_n_r, b_goal_r):  # b_n_r and b_goal_r are in the same shape
        for batch_ind, n_r in enumerate(b_n_r):
            goal_r = b_goal_r[batch_ind]
            self.link_chart[n_r][goal_r] = 1

    def link_exist_exam(self, b_n_r, b_goal_r):  # b_n_r and b_goal_r are in the same shape
        exist_list = torch.zeros_like(b_n_r, dtype=torch.float)
        for batch_ind, n_r in enumerate(b_n_r):
            goal_r = b_goal_r[batch_ind]
            exist_list[batch_ind] = self.link_chart[n_r][goal_r]
        return exist_list

    def find_neighbour_and_rel(self, e_ind, e_r=None, e_t=None):  # 不能包含训练集的准确实体
        rel_entity_list = copy.deepcopy(self.neighbor_data[e_ind, :])
        neighbour_id_list = torch.tensor([], dtype=torch.int64, device=self.device)
        relation_id_list = torch.tensor([], dtype=torch.int64, device=self.device)

        if e_r is not None:
            e_r = int(e_r)
        for rel_ind, neighbour_list in enumerate(rel_entity_list):
            if neighbour_list is not None:
                if rel_ind != e_r:
                    neighbour_list = torch.tensor(neighbour_list, device=self.device).to(torch.long)
                    rel_ind = torch.tensor(list([rel_ind]), device=self.device).to(torch.long)

                    rel_ind_like_neighbour_list = rel_ind.expand_as(neighbour_list)

                    neighbour_id_list = torch.cat((neighbour_id_list, neighbour_list), dim=0)
                    relation_id_list = torch.cat((relation_id_list, rel_ind_like_neighbour_list.view(-1)), dim=0)
                else:
                    neighbour_list.remove(float(e_t))  # 去除再训练集内的预测点
                    neighbour_list = torch.tensor(neighbour_list, device=self.device).to(torch.long)
                    rel_ind = torch.tensor(list([rel_ind]), device=self.device).to(torch.long)

                    rel_ind_like_neighbour_list = rel_ind.expand_as(neighbour_list)

                    neighbour_id_list = torch.cat((neighbour_id_list, neighbour_list), dim=0)
                    relation_id_list = torch.cat((relation_id_list, rel_ind_like_neighbour_list.view(-1)), dim=0)
            else:
                continue
        # 每个实体都有一个为自身的邻居，且关系为is
        neighbour_list = e_ind.view(-1, 1)
        rel_ind = torch.tensor(list([self.relation_cnt * 2]), device=self.device).to(torch.long)

        neighbour_id_list = torch.cat((neighbour_id_list, neighbour_list.view(-1)), dim=0)
        relation_id_list = torch.cat((relation_id_list, rel_ind.view(-1)), dim=0)

        return None, None, neighbour_id_list, relation_id_list

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector/scalar
        return unitized_vector

    def attention_based_infer(self, n_n, n_r, h, r):  # 用于合并所有路径得到的实体
        predict_x = self.single_infer.single_infer2entity(n_n, n_r, h, r)
        # infer_confidence = self.confidence_exam(n_r, r)
        predict_attention = self.attention(n_n, n_r, h, r)
        # y = self.logic_semantic_decision(predict_attention, infer_confidence, predict_x)
        y = torch.mm(predict_attention.T, predict_x)
        return y

    def score_based_infer(self, n_n, n_r, h, r):  # 用于合并所有路径得到的实体
        predict_x = self.single_infer.single_infer2entity(n_n, n_r, h, r)
        predict_attention = self.attention(n_n, n_r, h, r)
        predict_x = predict_x*predict_attention.view(-1, 1).expand_as(predict_x)
        # y = self.logic_semantic_decision(predict_attention, infer_confidence, predict_x)
        y, _ = torch.max(predict_x, dim=0)
        return y.view(1, -1)

    def select_attention_infer(self, n_n, n_r, h, r):  # 用于合并所有路径得到的实体
        predict_x = self.single_infer.single_infer2entity(n_n, n_r, h, r)
        # logic_attention = self.abs_attention(n_r, r)
        predict_attention = self.attention(n_n, n_r, h, r)
        ind = torch.argmax(predict_attention, dim=0)
        y = predict_x[ind, :]
        return y.view(1, -1)

    def train_attention_by_loss(self, n_n, n_r, h, r, t):  # 用于合并所有路径得到的实体
        predict_x = self.single_infer.single_infer2entity(n_n, n_r, h, r)
        _, index = torch.sort(predict_x, 1, descending=True)
        ranks = torch.zeros_like(t, dtype=torch.long, device=self.device)
        for ind, t0 in enumerate(t):
            logic_ind = index[ind, :] == t0
            rank = torch.nonzero(logic_ind == 1).squeeze()
            ranks[ind] = rank
        _, ind = torch.min(ranks, dim=0)
        goal_attention = torch.zeros_like(ranks, dtype=torch.float32, device=self.device)
        goal_attention[ind] = 1.0
        predict_attention = self.attention(n_n, n_r, h, r)
        attention_loss = self.attention_loss(predict_attention.view(1, -1), goal_attention.view(1, -1))
        return attention_loss

    def train_attention_by_rank(self, n_n, n_r, h, r, t):  # 用于合并所有路径得到的实体
        predict_x = self.single_infer.single_infer2entity(n_n, n_r, h, r)
        _, index = torch.sort(predict_x, 1, descending=True)
        rank_score = torch.zeros_like(t, dtype=torch.float32, device=self.device)
        for ind, t0 in enumerate(t):
            logic_ind = index[ind, :] == t0
            rank = torch.nonzero(logic_ind == 1).squeeze()
            rank_score[ind] = 1/(rank+1)
        predict_attention = self.attention(n_n, n_r, h, r)
        attention_loss = self.attention_loss(predict_attention.view(1, -1), rank_score.view(1, -1))
        return attention_loss

    def dirac_loss_list(self, predict_x, goal_ind):
        loss_tensor = torch.zeros_like(goal_ind, dtype=torch.float32, device=self.device)
        for x_ind, x in enumerate(predict_x):
            loss_tensor[x_ind] = self.loss(x.view(1, -1), goal_ind[x_ind])

        return loss_tensor

    '''def attention_based_infer(self, n_n, n_r, h, r, nl):  # 用于合并所有路径得到的实体
        predict_x = self.single_infer.single_infer2entity(n_n, n_r, h, r)
        survive_ratio = torch.ones([predict_x.shape[0], self.entity_cnt], device=self.device, dtype=torch.float32)
        if nl.view(-1).shape[0] > 0:
            survive_ratio[:, nl] = 0.0

        predict_survive = predict_x*survive_ratio
        # infer_confidence = self.confidence_exam(n_r, r)
        predict_attention = self.attention(n_n, n_r, h, r)
        # y = self.logic_semantic_decision(predict_attention, infer_confidence, predict_x)
        y = torch.mm(predict_attention.T, predict_survive)
        return y'''
    def y_ratio(self, nl):
        survive_ratio = torch.ones([1, self.entity_cnt], device=self.device, dtype=torch.float32)
        if nl.view(-1).shape[0] > 0:
            survive_ratio[:, nl] = 0.0
        return survive_ratio

    def select_based_infer(self, n_n, n_r, h, r):  # 用于合并所有路径得到的实体
        _, predict_y = self.single_infer.pre_train(n_n, n_r, h, r)
        select_ind = self.select_by_confidence(predict_y)
        select_n_n = n_n[select_ind].view(-1)
        select_n_r = n_r[select_ind].view(-1)
        select_h = h[select_ind].view(-1)
        select_r = r[select_ind].view(-1)

        predict_x = self.single_infer.single_infer2entity(select_n_n, select_n_r, select_h, select_r)
        predict_attention = self.attention(select_n_n, select_n_r, select_h, select_r)
        y = self.logic_semantic_decision(predict_attention, predict_x)
        return y

    def select_single_infer(self, n_n, n_r, h, r):
        _, predict_y = self.single_infer.pre_train(n_n, n_r, h, r)
        select_ind = self.select_by_confidence(predict_y)
        select_n_n = n_n[select_ind].view(-1)
        select_n_r = n_r[select_ind].view(-1)
        select_h = h[select_ind].view(-1)
        select_r = r[select_ind].view(-1)

        predict_x = self.single_infer.single_infer2entity(select_n_n, select_n_r, select_h, select_r)
        predict_attention = self.attention(select_n_n, select_n_r, select_h, select_r)
        y = self.logic_semantic_decision(predict_attention, predict_x)
        return y

    def select_by_confidence(self, predict_y):
        batch_max_y, _ = torch.max(predict_y, dim=1)
        temp_y = batch_max_y >= self.confidence_relu
        selected_ind = torch.nonzero(temp_y.view(-1) == 1).squeeze().view(-1)
        if selected_ind.shape[0] == 0:
            _, ind = torch.max(batch_max_y, dim=0)
            return ind
        elif selected_ind.shape[0] > 0:
            return selected_ind

    def logic_semantic_decision(self, attention, predict_x):
        temp_attention = attention >= self.logic_relu
        attention_ind = torch.nonzero(temp_attention.view(-1) == 1).squeeze().view(-1)
        if attention_ind.shape[0] > 0:
            return torch.mean(predict_x[attention_ind, :], dim=0)
        else:
            return torch.mm(attention.T, predict_x)

    def ignore_or_add_neighbour_randomly(self, selected_neighbour_ind, batch_t):
        '''if batch_t is None:  # 如果处于预测阶段，则不需要进行随机遮挡邻居操作
            return selected_neighbour_ind

        selected_neighbour_ind = selected_neighbour_ind.view(-1)
        survived_neighbour_ind = torch.tensor([], device=self.device, dtype=torch.int64)
        for times, neighbour_ind in enumerate(selected_neighbour_ind):
            survive_flag = torch.randint(0, 6, (1,))
            if survive_flag:
                survived_neighbour_ind = torch.cat((survived_neighbour_ind, neighbour_ind.view(-1)), dim=0)
        if survived_neighbour_ind.shape[0] == 0:
            survived_neighbour_ind = self.ignore_or_add_neighbour_randomly(selected_neighbour_ind, batch_t)
        return survived_neighbour_ind'''
        return selected_neighbour_ind

    def batch_multi_infer_model(self, batch_h, batch_r, batch_t):  # multi predict train
        batch_y = torch.tensor([], dtype=torch.float32, device=self.device)
        for batch_ind, head_id in enumerate(batch_h):
            rel_id = batch_r[batch_ind]
            if batch_t is None:  # 预测阶段获得所有邻居
                _, _, neighbour_id_list, neighbour_rel_id_list = self.find_neighbour_and_rel(head_id)
            else:  # 训练阶段排除t实体的连接
                _, _, neighbour_id_list, neighbour_rel_id_list = self.find_neighbour_and_rel(head_id, rel_id,
                                                                                             batch_t[batch_ind])
            rel_id_list = rel_id.expand_as(neighbour_rel_id_list)
            rel_attention = self.link_exist_exam(neighbour_rel_id_list, rel_id_list)

            rel_attention = rel_attention.view(-1)

            selected_path_list = rel_attention >= torch.max(rel_attention) - 0.2
            selected_neighbour_ind = torch.nonzero(selected_path_list == 1).squeeze()

            selected_neighbour_ind = self.ignore_or_add_neighbour_randomly(selected_neighbour_ind, batch_t)

            selected_neighbour_ind = selected_neighbour_ind.view(-1)

            batch_n = neighbour_id_list[selected_neighbour_ind].view(-1, 1)
            batch_n_r = neighbour_rel_id_list[selected_neighbour_ind].view(-1, 1)
            batch_h_ = head_id.expand_as(batch_n).view(-1, 1)
            batch_r_ = rel_id.expand_as(batch_n).view(-1, 1)

            batch_y = torch.cat((batch_y, self.attention_based_infer(batch_n, batch_n_r, batch_h_, batch_r_)), dim=0)

        return batch_y

    def train_batch_multi_infer_model(self, batch_h, batch_r, batch_t):  # multi predict train
        batch_loss = torch.zeros([batch_h.shape[0], 1], device=self.device, dtype=torch.float32)
        for batch_ind, head_id in enumerate(batch_h):
            rel_id = batch_r[batch_ind]
            if batch_t is None:  # 预测阶段获得所有邻居
                _, _, neighbour_id_list, neighbour_rel_id_list = self.find_neighbour_and_rel(head_id)
            else:  # 训练阶段排除t实体的连接
                _, _, neighbour_id_list, neighbour_rel_id_list = self.find_neighbour_and_rel(head_id, rel_id,
                                                                                             batch_t[batch_ind])
            rel_id_list = rel_id.expand_as(neighbour_rel_id_list)
            rel_attention = self.link_exist_exam(neighbour_rel_id_list, rel_id_list)

            rel_attention = rel_attention.view(-1)

            selected_path_list = rel_attention >= torch.max(rel_attention) - 0.2
            selected_neighbour_ind = torch.nonzero(selected_path_list == 1).squeeze()

            selected_neighbour_ind = self.ignore_or_add_neighbour_randomly(selected_neighbour_ind, batch_t)

            selected_neighbour_ind = selected_neighbour_ind.view(-1)

            batch_n = neighbour_id_list[selected_neighbour_ind]
            batch_n_r = neighbour_rel_id_list[selected_neighbour_ind]
            batch_h_ = head_id.expand_as(batch_n)
            batch_r_ = rel_id.expand_as(batch_n)
            batch_t_ = batch_t[batch_ind].expand_as(batch_n)

            batch_loss[batch_ind, :] = self.train_attention_by_loss(batch_n, batch_n_r, batch_h_, batch_r_, batch_t_)

        loss = torch.sum(batch_loss)/batch_h.shape[0]
        return loss, None

    def forward(self, batch_h, batch_r, batch_t=None):
        x = self.batch_multi_infer_model(batch_h, batch_r, batch_t)
        # x = F.relu(x)
        x = torch.mm(x, self.single_infer.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        # x += self.attention.e.expand_as(x)
        y = torch.sigmoid(x)
        return self.loss(y, batch_t), y


class LS_ConvE(BaseModel):
    def __init__(self, config):
        super(LS_ConvE, self).__init__(config)
        self.device = config.get('device')

        self.single_infer = single_infer(config)
        self.multi_infer = multi_infer(config, self.single_infer)

    def forward(self, batch_h, batch_r, batch_t=None):
        loss, y = self.multi_infer(batch_h, batch_r, batch_t)
        return loss, y


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


