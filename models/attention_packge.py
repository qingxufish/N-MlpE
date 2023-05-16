import torch
import torch.nn as nn
from .BaseModel import BaseModel
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F


class SPP(torch.nn.Module):
    def __init__(self, out_side):
        super(SPP, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out = None
        for n in self.out_side:
            max_pool = torch.nn.AdaptiveAvgPool3d([n, n, n])
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out


class SPP_attention_layer(BaseModel):
    def __init__(self, config):
        super(SPP_attention_layer, self).__init__(config)
        self.device = config.get('device')
        kwargs = config.get('Conv_attention_params')
        self.attention_w_out_dim = kwargs.get('attention_w_out_dim')
        self.emb_dim = kwargs.get('emb_dim')

        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.reshape = kwargs.get('reshape')
        self.kernel_size = kwargs.get('conv_kernel_size')
        self.stride = kwargs.get('stride')
        # convolution layer, in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=0
        self.conv1 = torch.nn.Conv2d(1, self.conv_out_channels, self.kernel_size, 1, 0, bias=kwargs.get('use_bias'))
        filtered_h = (self.reshape[0] * 2 - self.kernel_size[0]) // self.stride + 1
        filtered_w = (self.reshape[1] - self.kernel_size[1]) // self.stride + 1
        fc_length = self.conv_out_channels * filtered_h * filtered_w

        self.fc1 = torch.nn.Linear(fc_length, 1)
        self.w = torch.nn.Parameter(torch.Tensor(self.emb_dim, self.attention_w_out_dim))
        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        self.softmax = torch.nn.Softmax(0)
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, n_r, goal_r):  # 利用attention计算合并各条路径的计算结果
        n_size = n_r.shape[0]
        n_r_w = torch.mm(n_r, self.w)
        goal_r_w = torch.mm(goal_r, self.w)

        r1 = n_r_w.view(-1, 1, *self.reshape)
        r2 = goal_r_w.view(-1, 1, *self.reshape)
        stacked_inputs = torch.cat([r1, r2], 2)  # (batch_size, 1, 2*emb_dim1, emb_dim2)

        x = self.conv1(stacked_inputs)
        x = x.view(n_size, -1)

        x = self.leakyrelu(x)
        x = self.fc1(x)

        x = self.softmax(x)
        return x


class ConvR_attention(BaseModel):
    def __init__(self, config):
        super(ConvR_attention, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('attention_params')
        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.reshape = kwargs.get('reshape')
        self.kernel_size = kwargs.get('conv_kernel_size')
        self.stride = kwargs.get('stride')
        self.emb_dim = {
            'entity': kwargs.get('emb_dim'),
            'relation': self.conv_out_channels * self.kernel_size[0] * self.kernel_size[1]-kwargs.get('emb_dim')
        }
        assert self.emb_dim['entity'] == self.reshape[0] * self.reshape[1] - self.emb_dim['relation']
        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim['entity'])
        self.R = torch.nn.Embedding(self.relation_cnt * 2+1, self.emb_dim['relation'])
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.register_parameter('b', Parameter(torch.zeros(self.entity_cnt)))
        self.register_parameter('c', Parameter(torch.ones(1)))
        self.filtered = [(self.reshape[0] - self.kernel_size[0]) // self.stride + 1,
                         (self.reshape[1] - self.kernel_size[1]) // self.stride + 1]
        fc_length = self.conv_out_channels * self.filtered[0] * self.filtered[1]
        self.fc = torch.nn.Linear(fc_length, 1)
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

    def forward(self, batch_h_, batch_r_, batch_h, batch_r):
        batch_size = batch_h.size(0)

        e0 = self.E(batch_h_)
        r0 = self.R(batch_r_)
        e0r0 = torch.cat((e0, r0), dim=2).view(-1, 1, *self.reshape)
        e0r0 = e0r0.view(1, -1, *self.reshape)

        e = self.E(batch_h)
        r = self.R(batch_r)
        er = torch.cat((e, r), dim=2)
        er = er.view(-1, 1, *self.kernel_size)

        x = torch.conv2d(e0r0, er, groups=batch_size)
        x = x.view(batch_size, self.conv_out_channels, *self.filtered)
        x = F.leaky_relu(x, 0.1)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class Conv_attention_layer(BaseModel):
    def __init__(self, config):
        super(Conv_attention_layer, self).__init__(config)
        self.device = config.get('device')
        kwargs = config.get('Conv_attention_params')
        self.attention_w_out_dim = kwargs.get('attention_w_out_dim')
        self.emb_dim = kwargs.get('emb_dim')

        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.reshape = kwargs.get('reshape')
        self.kernel_size = kwargs.get('conv_kernel_size')
        self.stride = kwargs.get('stride')
        # convolution layer, in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=0
        self.conv1 = torch.nn.Conv2d(1, self.conv_out_channels, self.kernel_size, 1, 0, bias=kwargs.get('use_bias'))
        filtered_h = (self.reshape[0] * 2 - self.kernel_size[0]) // self.stride + 1
        filtered_w = (self.reshape[1] - self.kernel_size[1]) // self.stride + 1
        fc_length = self.conv_out_channels * filtered_h * filtered_w

        self.fc1 = torch.nn.Linear(fc_length, 1)
        self.w = torch.nn.Parameter(torch.Tensor(self.emb_dim, self.attention_w_out_dim))
        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        self.softmax = torch.nn.Softmax(0)
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, n_r, goal_r):  # 利用attention计算合并各条路径的计算结果
        n_size = n_r.shape[0]
        n_r_w = torch.mm(n_r, self.w)
        goal_r_w = torch.mm(goal_r, self.w)

        r1 = n_r_w.view(-1, 1, *self.reshape)
        r2 = goal_r_w.view(-1, 1, *self.reshape)
        stacked_inputs = torch.cat([r1, r2], 2)  # (batch_size, 1, 2*emb_dim1, emb_dim2)

        x = self.conv1(stacked_inputs)
        x = x.view(n_size, -1)

        x = self.leakyrelu(x)
        x = self.fc1(x)

        x = self.softmax(x)
        return x


class attention_layer(BaseModel):
    def __init__(self, config):
        super(attention_layer, self).__init__(config)
        self.device = config.get('device')
        kwargs = config.get('attention_params')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        self.emb_dim = kwargs.get('attention_emb_dim')
        self.self_attention = SelfAttention(num_attention_heads=4, input_size=self.emb_dim*4,
                                            hidden_size=self.emb_dim*4, hidden_dropout_prob=0.1)

        self.encoder1 = torch.nn.TransformerEncoderLayer(self.emb_dim * 4, 32, self.emb_dim * 4, 0.0)
        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim)
        self.R = torch.nn.Embedding(self.relation_cnt * 2 + 1, self.emb_dim)  # 反向为另一种关系
        self.register_parameter('a', Parameter(torch.ones(1)))
        self.register_parameter('w1', Parameter(torch.ones(self.emb_dim * 4, self.emb_dim * 4)))
        self.fc = torch.nn.Linear(self.emb_dim * 4, 1)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        self.softmax = torch.nn.Softmax(0)
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.R.weight.data)
        torch.nn.init.xavier_normal_(self.E.weight.data)


    def forward(self, n, n_r, h, goal_r):  # 利用attention计算合并各条路径的计算结果
        n_tensor = self.E(n)
        n_r_tensor = self.R(n_r)
        h_tensor = self.E(h)
        g_r_tensor = self.R(goal_r)

        x = torch.cat((n_tensor, n_r_tensor, h_tensor, g_r_tensor), dim=1)
        x = x.view(-1, self.emb_dim*4)
        x = self.encoder1(x.unsqueeze(0)).squeeze()
        # x = self.self_attention(x.unsqueeze(0)).squeeze()
        x = self.leakyrelu(x)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x.view(-1, 1)


class abs_attention(BaseModel):
    def __init__(self, config):
        super(abs_attention, self).__init__(config)
        self.device = config.get('device')
        kwargs = config.get('attention_params')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        self.emb_dim = kwargs.get('attention_emb_dim')
        self.R = torch.nn.Embedding(self.relation_cnt * 2 + 1, self.emb_dim)  # 反向为另一种关系

        self.fc = torch.nn.Linear(self.emb_dim * 2, 1)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        self.softmax = torch.nn.Softmax(0)
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.R.weight.data)

    def forward(self, n_r, goal_r):  # 利用attention计算合并各条路径的计算结果
        n_r_tensor = self.R(n_r.view(-1))
        g_r_tensor = self.R(goal_r.view(-1))

        x = torch.cat([n_r_tensor, g_r_tensor], dim=1)
        x = self.leakyrelu(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        #b = x-self.threshord
        #b = torch.relu(b)
        #x = torch.sigmoid(x)

        return x


class max_attention_layer(BaseModel):
    def __init__(self, config):
        super(max_attention_layer, self).__init__(config)
        self.device = config.get('device')
        kwargs = config.get('model_hyper_params')
        self.attention_w_out_dim = kwargs.get('attention_w_out_dim')
        self.emb_dim = kwargs.get('emb_dim')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim)
        self.R = torch.nn.Embedding(self.relation_cnt * 2 + 1, self.emb_dim)  # 反向为另一种关系

        self.fc = torch.nn.Linear(self.emb_dim * 4, 1)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        self.threshord = torch.nn.Parameter(torch.Tensor(1))
        self.softmax = torch.nn.Softmax(0)
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.R.weight.data)
        torch.nn.init.xavier_normal_(self.E.weight.data)

    def forward(self, n, n_r, h, goal_r):  # 利用attention计算合并各条路径的计算结果
        n_tensor = self.E(n)
        n_r_tensor = self.R(n_r)
        h_tensor = self.E(h)
        g_r_tensor = self.R(goal_r)

        x = torch.cat((n_tensor, n_r_tensor, h_tensor, g_r_tensor), dim=1)
        x = self.leakyrelu(x)
        x = self.fc(x)
        limit, _ = torch.max(x, dim=0)
        x = torch.heaviside(x - limit, torch.ones_like(x, dtype=torch.float32, device=self.device))

        return x


class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)  # -> (img_size, img_size)
        patch_size = (patch_size, patch_size)  # -> (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 假设采取默认参数
        x = self.proj(x) # 出来的是(N, 96, 224/4, 224/4)
        x = torch.flatten(x, 2) # 把HW维展开，(N, 96, 56*56)
        x = torch.transpose(x, 1, 2)  # 把通道维放到最后 (N, 56*56, 96)
        if self.norm is not None:
            x = self.norm(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        #hidden_states = self.out_dropout(hidden_states)
        #hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class KQ_Attention(nn.Module):

    def __init__(self, config):
        super(KQ_Attention, self).__init__()
        kwargs = config.get('model_hyper_params')
        self.ent_dim = kwargs.get('entity_dim')
        self.rel_dim = kwargs.get('relation_dim')

        key_input_dim = (self.ent_dim+self.rel_dim)*2
        query_input_dim = self.ent_dim+self.rel_dim

        self.key_encoder = torch.nn.Linear(key_input_dim, key_input_dim)
        self.query_encoder = torch.nn.Linear(query_input_dim, query_input_dim)

        self.key_fc = torch.nn.Linear(key_input_dim, self.ent_dim)
        self.query_fc = torch.nn.Linear(query_input_dim, self.ent_dim)

        self.softmax = torch.nn.Softmax(dim=0)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)

    def forward(self, n_n, n_r, n_h, r):
        key_input = torch.cat((n_n, n_r, n_h, r), dim=1)
        key_input = self.leakyrelu(self.key_encoder(key_input))
        key = self.key_fc(key_input)

        query_input = torch.cat((n_h, r), dim=1)
        query_input = self.leakyrelu(self.query_encoder(query_input))
        query = self.query_fc(query_input)

        score = torch.mm(key.T, query)

        return score


class SA_attention(BaseModel):
    def __init__(self, config):
        super(SA_attention, self).__init__(config)
        kwargs = config.get('model_hyper_params')
        self.ent_dim = kwargs.get('entity_dim')
        self.rel_dim = kwargs.get('relation_dim')
        input_dim = (self.ent_dim+self.rel_dim)*2

        attention_config = config.get('attention_config')
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=attention_config.get('head_num'))
        self.mix = nn.TransformerEncoder(encoder_layer, num_layers=attention_config.get('encoder_num'))

        self.fc = torch.nn.Linear(input_dim, 1)
        self.softmax = torch.nn.Softmax(dim=0)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)

    def forward(self,  n_n, n_n_r, n_h, n_r):
        input = torch.cat((n_n, n_n_r, n_h, n_r), dim=1).unsqueeze(0)
        input = self.mix(input)
        input = input.squeeze(0)

        input = self.leakyrelu(input)
        y = self.softmax(self.fc(input))
        return y


class FC_attention(nn.Module):
    def __init__(self, config):
        super(FC_attention, self).__init__()
        kwargs = config.get('model_hyper_params')
        self.ent_dim = kwargs.get('entity_dim')
        self.rel_dim = kwargs.get('relation_dim')

        key_input_dim = self.ent_dim+self.rel_dim
        query_input_dim = self.ent_dim+self.rel_dim

        self.key_encoder = torch.nn.Linear(key_input_dim, key_input_dim)
        self.query_encoder = torch.nn.Linear(query_input_dim, query_input_dim)

        self.key_fc = torch.nn.Linear(key_input_dim, self.ent_dim)
        self.query_fc = torch.nn.Linear(query_input_dim, self.ent_dim)

        self.softmax = torch.nn.Softmax(dim=0)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)

    def forward(self, n_n, n_r, n_h, r):
        key_input = torch.cat((n_n, n_r), dim=1)
        key_input = self.leakyrelu(self.key_encoder(key_input))
        key = self.key_fc(key_input)

        query_input = torch.cat((n_h, r), dim=1)
        query_input = self.leakyrelu(self.query_encoder(query_input))
        query = self.query_fc(query_input)

        score = torch.sum(torch.mul(key, query), dim=1).view(-1, 1)

        return score

class ALL_attention(nn.Module):
    def __init__(self, config):
        super(ALL_attention, self).__init__()
        kwargs = config.get('model_hyper_params')
        self.ent_dim = kwargs.get('entity_dim')
        self.rel_dim = kwargs.get('relation_dim')

        input_dim = (self.ent_dim+self.rel_dim)*2

        self.encoder = torch.nn.Linear(input_dim, input_dim*2)
        self.fc = torch.nn.Linear(input_dim*2, 1)

        self.leakyrelu = torch.nn.LeakyReLU(0.1)

    def forward(self, n_n, n_r, n_h, r):
        input = torch.cat((n_n, n_r, n_h, r), dim=1)
        input = self.leakyrelu(self.encoder(input))
        score = self.fc(input)

        return score


