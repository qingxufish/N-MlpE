import copy
import pickle

import networkx as nx
import torch
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel

class triples_Transformer_MlpE(BaseModel):
    def __init__(self, config):
        super(triples_Transformer_MlpE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.ent_dim = kwargs.get('entity_dim')
        self.rel_dim = kwargs.get('relation_dim')
        self.k = kwargs.get('k')

        self.E = torch.nn.Embedding(self.entity_cnt+1, self.ent_dim)  # 最后一个为特殊token
        self.R = torch.nn.Embedding(self.relation_cnt*2, self.rel_dim)  # 反向为另一种关系
        self.init()

        path_kwargs = config.get('pathEmbParams')
        self.nhead = path_kwargs.get('head')
        self.layer_num = path_kwargs.get('layer_num')
        encoder_layer = torch.nn.TransformerEncoderLayer(self.ent_dim*2 + self.rel_dim, nhead=self.nhead)
        self.pathEmbLayer = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.layer_num)

        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn2 = torch.nn.BatchNorm1d(1)
        self.register_parameter('c', Parameter(torch.ones(1)))

        message_model_kwarg = config.get('MlpE_params')
        self.feature_num = message_model_kwarg.get('feature_num')

        self.encoder = torch.nn.Linear(self.ent_dim*2 + self.rel_dim, self.feature_num)
        self.activate = torch.nn.LeakyReLU(0.1)
        self.decoder = torch.nn.Linear(self.feature_num, self.ent_dim)

        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)

        try:
            f = open(kwargs.get('subEdgeDic_path'), 'rb')
            self.subEdgeDic = pickle.load(f)
            self.fileFlag = 1
            f.close()
        except FileNotFoundError:
            self.fileFlag = 0
            self.G = nx.MultiDiGraph()  # 初始化训练集构成的图，方便构建n阶子图
            self.subEdgeDic = {}  # 存储子图字典，索引为(h,r),返回子图实例

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector / scalar
        return unitized_vector

    def genSubEdgeFile(self, batch_h, batch_r, batch_t):
        if batch_t is not None:
            for batch_ind, head_ind in enumerate(batch_h):
                rel_ind = batch_r[batch_ind]
                tail_ind = batch_t[batch_ind]
                self.updateDic(head_ind, rel_ind, tail_ind)
        else:
            for batch_ind, head_ind in enumerate(batch_h):
                rel_ind = batch_r[batch_ind]
                self.updateDic(head_ind, rel_ind)

    def updateDic(self,e_h, e_r, e_t=None):  # 不能包含训练集的准确实体
        if e_t is not None:  # 训练时,更新字典
            e_h = e_h.tolist()
            e_r = e_r.tolist()
            e_t = e_t.tolist()

            subgraph = nx.ego_graph(self.G, n=e_h, radius=self.k)  # 生成以节点e为中心的k阶子图
            subgraph.remove_edge(e_h, e_t, key=e_r)  # 去除训练标签三元组
            edges = subgraph.edges.data(keys=True)  # 获取边的数据
            sequence = torch.tensor([item[:3] for item in list(edges)])
            lastItem = torch.cat((torch.tensor(e_h).view(1,-1),
                                  torch.tensor(self.entity_cnt).view(1,-1),
                                  torch.tensor(e_r).view(1,-1)), dim=1)
            inputSequence = torch.cat((sequence, lastItem), dim=0)

            self.subEdgeDic.update({(e_h, e_r): inputSequence})

        else:  # 预测时只需要按照中心点找到子图即可，不需要删除某些边，更新字典
            e_h = e_h.tolist()
            e_r = e_r.tolist()
            subgraph = nx.ego_graph(self.G, n=e_h, radius=self.k)  # 生成以节点e为中心的k阶子图
            edges = subgraph.edges.data(keys=True)  # 获取边的数据

            sequence = torch.tensor([item[:3] for item in list(edges)])
            lastItem = torch.cat((torch.tensor(e_h).view(1, -1),
                                  torch.tensor(self.entity_cnt).view(1, -1),
                                  torch.tensor(e_r).view(1, -1)), dim=1)
            inputSequence = torch.cat((sequence, lastItem), dim=0)

            self.subEdgeDic.update({(e_h, e_r): inputSequence})

    def singleSequence(self, e_h, e_r):
        try:
            hrtList = self.subEdgeDic[(e_h.tolist(), e_r.tolist())].to(self.device)
            sequence = torch.cat((self.E(hrtList[:, 0]),  self.R(hrtList[:, 2]), self.E(hrtList[:, 1])), dim=1)
        except RuntimeError:
            hrtList = self.subEdgeDic[(e_h.tolist(), e_r.tolist())].to(torch.long).to(self.device)
            sequence = torch.cat((self.E(hrtList[:, 0]), self.R(hrtList[:, 2]), self.E(hrtList[:, 1])), dim=1)

        result = self.pathEmbLayer(sequence)[-1,:].view(1,-1)
        return result

    def genSequenceByDic(self, batch_h, batch_r):
        batchResult = torch.tensor([]).to(self.device)
        for batch_ind, head_ind in enumerate(batch_h):
            rel_ind = batch_r[batch_ind]
            batchResult = torch.cat((batchResult, self.singleSequence(head_ind, rel_ind)), dim=0)

        return batchResult


    def forward(self, batch_h, batch_r, batch_t=None):
        if self.fileFlag:
            x = self.genSequenceByDic(batch_h, batch_r)
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
            x = torch.mm(x, entities_embedding.transpose(1, 0)[:,:-1])  # *self.c  # c is important
            # x += self.b.expand_as(x)  # deletable
            y = torch.sigmoid(x)
            return self.loss(y, batch_t), y
        else:
            self.genSubEdgeFile(batch_h, batch_r, batch_t)
            batch_t = None
            #  假数据，为了再验证的时候能够顺利运行，没有其他意义
            x = torch.mm(self.E(batch_h), self.E.weight.transpose(1, 0)[:,:-1])
            y = torch.sigmoid(x)
            return self.loss(1, batch_t), y

class ConvELoss(BaseModel):
    def __init__(self, device, label_smoothing, entity_cnt):
        super().__init__()
        self.device = device
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.label_smoothing = label_smoothing
        self.entity_cnt = entity_cnt

    def forward(self, batch_p, batch_t=None):
        loss = torch.tensor(1)  # 为了让train中的loss.backward()可以执行而设计，没有其他意义
        if batch_t is not None:
            batch_size = batch_p.shape[0]
            batch_e = torch.zeros(batch_size, self.entity_cnt).to(self.device).scatter_(1, batch_t.view(-1, 1), 1)
            batch_e = (1.0 - self.label_smoothing) * batch_e + self.label_smoothing / self.entity_cnt
            loss = self.loss(batch_p, batch_e) / batch_size
        return loss

