import torch
import logging
from .ConvE import ConvE
from .ConvR import ConvR
from .ConvE_for_GCN import ConvE_for_GCN
from .double_ConvE import double_ConvE
from .neighbour_attention_ConvE import neighbour_attention_ConvE
from .elect_attention_ConvE import elect_attention_ConvE
from .LS_ConvE import LS_ConvE
from .merge_neighbour_ConvE import merge_neighbour_ConvE
from .ConvKB import ConvKB
from torch.utils.data import DataLoader
from tqdm import tqdm
from .DIS_ConvE import DIS_ConvE
from .Original_ConvE import Original_ConvE
from .ALL_ConvE import ALL_ConvE
from .FCNN_ConvE import FCNN_ConvE
from .attention_ConvR import attention_ConvR
from .bais_ConvE import bais_ConvE
from .self_attention_ConvE import self_attention_ConvE
from .MlpE import MlpE
from .A_MlpE import A_MlpE
from .Milti_hops_MlpE import Multi_hops_MlpE
from .Hard_filter_multihop_MlpE import Hard_filter_multihop_MlpE
from .transformer_MlpE import transformer_MlpE
from .triples_Transformer_MlpE import triples_Transformer_MlpE
from .triples_Transformer_MlpE_without_files import triples_Transformer_MlpE_without_files

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_dict = {
    'ConvE': ConvE,
    'ConvR': ConvR,
    'ConvE_for_GCN': ConvE_for_GCN,
    'double_ConvE': double_ConvE,
    'neighbour_attention_ConvE': neighbour_attention_ConvE,
    'elect_attention_ConvE': elect_attention_ConvE,
    'LS_ConvE': LS_ConvE,
    'merge_neighbour_ConvE': merge_neighbour_ConvE,
    'ConvKB': ConvKB,
    'DIS_ConvE': DIS_ConvE,
    'Original_ConvE': Original_ConvE,
    'ALL_ConvE': ALL_ConvE,
    'FCNN_ConvE': FCNN_ConvE,
    'attention_ConvR': attention_ConvR,
    'bais_ConvE': bais_ConvE,
    'self_attention_ConvE': self_attention_ConvE,
    'MlpE': MlpE,
    'A_MlpE': A_MlpE,
    'Multi_hops_MlpE': Multi_hops_MlpE,
    'Hard_filter_multihop_MlpE': Hard_filter_multihop_MlpE,
    'transformer_MlpE' : transformer_MlpE,
    'triples_Transformer_MlpE' : triples_Transformer_MlpE,
    'triples_Transformer_MlpE_without_files' : triples_Transformer_MlpE_without_files
}


def init_model(config, exp_class):
    # initialize model
    device = config.get('device')
    if config.get('model_name') in model_dict:
        model = model_dict[config.get('model_name')].init_model(config)
    else:
        raise ValueError('Model not support: ' + config.get('model_name'))
    logging.info(model)

    '''
    # For simplicity, use DataParallel wrapper to use multiple GPUs.
    if device == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f'{torch.cuda.device_count()} GPUs are available. Let\'s use them.')
        model = torch.nn.DataParallel(model)
    '''
    try:
        model = load_link(exp_class, model)
        model = model.to(device)
        logging.info(f'model loaded on {device}')
    except AttributeError:
        model = model.to(device)
        logging.info(f'model loaded on {device}')

    try:
        model = load_G(exp_class, model, config)
        model = model.to(device)
        logging.info(f'model loaded on {device}')
    except AttributeError:
        model = model.to(device)
        logging.info(f'model loaded on {device}')

    param_count = 0
    for p in model.parameters():
        param_count += p.view(-1).size()[0]
    logging.info(f'model param is {param_count}')

    return model, device


def load_link(exp_class, model):
    train_data = DataLoader(exp_class.dataset.data['double_triplets_train'], exp_class.train_conf.get('batch_size'), drop_last=False)
    for batch_data in tqdm(train_data):
        n_r = batch_data[1]
        r = batch_data[3]
        model.multi_infer.memorize_link(n_r, r)  # 记录所有共同出现的关系
    return model

def load_G(exp_class, model, config):
    relcont = config['relation_cnt']
    train_data = exp_class.dataset.data['train']
    node_data = exp_class.dataset.data['entity']
    model.G.add_nodes_from(node_data)
    for batch_data in tqdm(train_data):
        model.G.add_edge(batch_data[0], batch_data[1], key=batch_data[2])
        model.G.add_edge(batch_data[1], batch_data[0], key=batch_data[2]+relcont)
    return model