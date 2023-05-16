import os
import json
import torch
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_json_config(config_path, args):
    logging.info(' Loading configuration '.center(100, '-'))
    if not os.path.exists(config_path):
        logging.warning(f'File {config_path} does not exist, empty list is returned.')
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config['GPU']:
        config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        config['device'] = torch.device('cpu')
    return config


def load_triples(file_path):
    tuples = []
    if not os.path.exists(file_path):
        logging.warning(f'File {file_path} does not exist, empty list is returned.')
    else:
        with open(file_path, 'r') as f:
            data = f.readlines()
            logging.info('%d triples loaded from %s.' % (len(data)-1, file_path))
            for line in data[1:]:
                record = line.strip().split(' ')
                tuples.append(tuple(map(int, record)))
    return tuples


def load_double_triples(file_path):
    tuples = []
    if not os.path.exists(file_path):
        logging.warning(f'File {file_path} does not exist, empty list is returned.')
    else:
        with open(file_path, 'r') as f:
            data = f.readlines()
            logging.info('%d triples loaded from %s.' % (len(data)-1, file_path))
            for line in data[1:]:
                record = line.strip().split('\t')
                tuples.append(tuple(map(int, record)))
    return tuples


def load_infer_triples(file_path):
    tuples = []
    if not os.path.exists(file_path):
        logging.warning(f'File {file_path} does not exist, empty list is returned.')
    else:
        with open(file_path, 'r') as f:
            data = f.readlines()
            logging.info('%d triples loaded from %s.' % (len(data)-1, file_path))
            for line in data[1:]:
                record = line.strip().split('\t')
                tuples.append(tuple(map(int, record)))

        triplets_list = list()
        temp_list = list()
        ind = 0
        for infer_triplet_ind, infer_triplet in enumerate(tuples):
            if tuples[ind][5] == infer_triplet[5]:
                temp_list.append(torch.tensor(infer_triplet[:5]))  # insert
            else:
                triplets_list.append({temp_list[0][3:5]:temp_list})
                ind = infer_triplet_ind
                temp_list = [torch.tensor(infer_triplet[:5])]  # empty temp_list
    return triplets_list


def load_ids(file_path):
    ids = []
    if not os.path.exists(file_path):
        logging.warning(f'File {file_path} does not exist, empty list is returned.')
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            logging.info('%d of entities/relations loaded from %s.' % (len(data)-1, file_path))
            for line in data[1:]:
                record = line.strip()
                try:
                    id = record.split('\t')[1]
                except IndexError:
                    id = record.split(' ')[-1]
                ids.append(int(id))
    return ids


def calc_goal_distribute(batch_h, batch_r, batch_t, goal_data, num_entities, num_relations):
    head_set = batch_h
    rel_set = batch_r
    tail_set = batch_t

    goal_dis = np.empty([rel_set.shape[0], num_entities])
    for triplet_id in range(int(rel_set.shape[0]/2)):
        head_id = head_set[triplet_id]
        rel_id = rel_set[triplet_id]
        tail_id = tail_set[triplet_id]

        head_rel_dis = torch.tensor(goal_data[head_id][rel_id])
        rel_tail_dis = torch.tensor(goal_data[tail_id][rel_id+num_relations])

        s2t_distribute = torch.zeros(num_entities)
        t2s_distribute = torch.zeros(num_entities)

        s2t_distribute = s2t_distribute.index_fill(0, head_rel_dis.to(torch.long), 1)
        t2s_distribute = t2s_distribute.index_fill(0, rel_tail_dis.to(torch.long), 1)

        goal_dis[triplet_id, :] = s2t_distribute.tolist()
        goal_dis[triplet_id+int(rel_set.shape[0]/2), :] = t2s_distribute.tolist()

    return goal_dis


def generate_goal_array(train_triplets, num_entities, num_relation):
    train_triplets = np.array(train_triplets)
    head_relation_triplets = torch.tensor(train_triplets[:, 0::2])
    relation_tail_triplets = torch.tensor(train_triplets[:, -1:-3:-1].copy())
    train_triplets = torch.tensor(train_triplets)

    goal_array = np.empty([num_entities, num_relation*2], dtype=list)  # 构建目标索引列表
    ind = 0
    for triplet in train_triplets:
        triplet = triplet.tolist()
        head_relation = torch.tensor(triplet[0::2])
        relation_tail = torch.tensor(triplet[-1:-3:-1].copy())
        triplet = torch.tensor(triplet)

        tail_index = torch.sum(head_relation_triplets == head_relation, dim=1)
        tail_index = torch.nonzero(tail_index == 2).squeeze()
        tail_index = train_triplets[tail_index, 1].view(-1).tolist()

        head_index = torch.sum(relation_tail_triplets == relation_tail, dim=1)
        head_index = torch.nonzero(head_index == 2).squeeze()
        head_index = train_triplets[head_index, 0].view(-1).tolist()

        goal_array[int(triplet[0])][int(triplet[2])] = tail_index
        goal_array[int(triplet[1])][int(triplet[2])+num_relation] = head_index
        ind = ind + 1
        print(ind)

    return goal_array
