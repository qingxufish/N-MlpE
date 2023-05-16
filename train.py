import networkx
import torch
import logging 
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_without_label(dataset, model, optimizer, device, batch_size=None):
    data = DataLoader(dataset.data['train'], batch_size, drop_last=False, shuffle=True)
    full_loss = []
    model.train()
    rel_cnt = model.relation_cnt
    for batch_data in tqdm(data):
        h = batch_data[0].to(device)
        t = batch_data[1].to(device)
        r = batch_data[2].to(device)
        r_ = batch_data[2].to(device) + rel_cnt
        optimizer.zero_grad()
        loss_to_tail, _ = model(h, r, t)  # (h,r,t)
        loss_to_head, _ = model(t, r_, h)  # (t,r',h)
        # loss = loss.mean()
        loss = (loss_to_tail + loss_to_head)/2
        loss.backward()
        optimizer.step()
        full_loss.append(loss.item())
    return full_loss


def train_double_triplets(dataset, model, optimizer, device, batch_size):
    train_data = DataLoader(dataset.data['double_triplets_train'], batch_size, drop_last=False)
    model.train()
    # train singel_infer_model
    for p in model.single_predict_model.parameters():
        p.requires_grad = True
    model.single_predict_model.train()
    for _ in range(0, 0):
        full_loss = []
        for batch_data in tqdm(train_data):
            n = batch_data[0].to(device)
            n_r = batch_data[1].to(device)
            h = batch_data[2].to(device)
            r = batch_data[3].to(device)
            t = batch_data[4].to(device)
            flag = batch_data[5].to(device)

            optimizer.zero_grad()
            # train single infer
            infer_loss, _ = model(batch_h=h, batch_r=r, batch_n=n, batch_n_r=n_r, batch_t=t, batch_flag=flag)  # (h,r,t)
            # loss = loss.mean()
            loss = infer_loss
            loss.backward()
            optimizer.step()
            full_loss.append(loss.item())
        print(np.mean(full_loss))

    infer_data = dataset.data['Infer_data']
    batch_ind_list = DataLoader(list(range(0, len(infer_data))), batch_size, drop_last=False)
    # frozen single_infer_model's parameters
    for p in model.single_predict_model.parameters():
        p.requires_grad = False
    # disable the dropout layers
    model.single_predict_model.eval()
    for _ in range(0, 0):
        full_loss = []
        for batch_list in tqdm(batch_ind_list):
            min_ind = batch_list[0]
            max_ind = batch_list[-1] + 1
            infer_data[min_ind: max_ind]

            optimizer.zero_grad()
            # train multi infer
            infer_loss, _ = model(batch_h=None, batch_r=None, batch_multi_infer=infer_data[min_ind: max_ind])  # (h,r,t)
            # loss = loss.mean()
            loss = infer_loss
            loss.backward()
            optimizer.step()
            full_loss.append(loss.item())
        print(np.mean(full_loss))
    return full_loss


def train_elect_attention_ConvE(dataset, model, optimizer, device, batch_size):
    train_data = DataLoader(dataset.data['double_triplets_train'], batch_size, drop_last=False)
    for _ in range(0, 10):
        full_loss = []
        for batch_data in tqdm(train_data):
            n = batch_data[0].to(device)
            n_r = batch_data[1].to(device)
            h = batch_data[2].to(device)
            r = batch_data[3].to(device)
            t = batch_data[4].to(device)
            flag = batch_data[5].to(device)

            optimizer.zero_grad()
            # train single infer
            infer_loss, _ = model(batch_h=h, batch_r=r, batch_n=n, batch_n_r=n_r, batch_t=t, batch_flag=flag)  # (h,r,t)
            # loss = loss.mean()
            loss = infer_loss
            loss.backward()
            optimizer.step()
            full_loss.append(loss.item())
        print(np.mean(full_loss))
    return full_loss

def train_LS_ConvE(dataset, model, optimizer, device, batch_size):
    pre_train_data = DataLoader(dataset.data['double_triplets_train'], batch_size, drop_last=False)
    if model.multi_infer.pre_train_flag:
        model.multi_infer.pre_train_flag = 0
        optimizer = torch.optim.Adam(model.single_infer.parameters(), lr=0.001)
        for _ in range(0, 1):
            full_loss = []
            for batch_data in tqdm(pre_train_data):
                n = batch_data[0].to(device)
                n_r = batch_data[1].to(device)
                h = batch_data[2].to(device)
                r = batch_data[3].to(device)
                t = batch_data[4].to(device)

                optimizer.zero_grad()
                # train single infer
                infer_loss, _ = model.single_infer.pre_train(batch_h=h, batch_r=r, batch_n=n, batch_n_r=n_r, batch_t=t)  # (h,r,t)
                # loss = loss.mean()
                loss = infer_loss
                loss.backward()
                optimizer.step()
                full_loss.append(loss.item())
            print(np.mean(full_loss))

    model.single_infer.eval()

    full_loss = []
    for batch_data in tqdm(pre_train_data):
        n = batch_data[0].to(device)
        n_r = batch_data[1].to(device)
        h = batch_data[2].to(device)
        r = batch_data[3].to(device)
        t = batch_data[4].to(device)
        # train single infer
        infer_loss, _ = model.single_infer.pre_train(batch_h=h, batch_r=r, batch_n=n, batch_n_r=n_r,
                                                         batch_t=t)  # (h,r,t)
        # loss = loss.mean()
        loss = infer_loss
        full_loss.append(loss.item())
    print(np.mean(full_loss))

    train_data = DataLoader(dataset.data['train'], batch_size, drop_last=False)
    for _ in range(0, 1):
        full_loss = []
        optimizer0 = torch.optim.Adam(model.multi_infer.parameters(), lr=0.0001)
        optimizer1 = torch.optim.Adam(model.multi_infer.attention.parameters(), lr=0.01)
        rel_cnt = model.multi_infer.relation_cnt
        for batch_data in tqdm(train_data):
            h = batch_data[0].to(device)
            t = batch_data[1].to(device)
            r = batch_data[2].to(device)
            r_ = batch_data[2].to(device) + rel_cnt
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            loss_to_tail, _ = model.multi_infer(h, r, t)  # (h,r,t)
            loss_to_head, _ = model.multi_infer(t, r_, h)  # (t,r',h)
            # loss = loss.mean()
            loss = (loss_to_tail + loss_to_head)/2
            loss.backward()
            optimizer0.step()
            optimizer1.step()
            full_loss.append(loss.item())
    return full_loss

def train_ALL_ConvE(dataset, model, optimizer, device, batch_size):
    train_data = DataLoader(dataset.data['train'], batch_size, drop_last=False)
    for _ in range(0, 1):
        full_loss = []
        rel_cnt = model.relation_cnt
        for batch_data in tqdm(train_data):
            h = batch_data[0].to(device)
            t = batch_data[1].to(device)
            r = batch_data[2].to(device)
            r_ = batch_data[2].to(device) + rel_cnt
            optimizer.zero_grad()
            loss_to_tail, _ = model(h, r, t)  # (h,r,t)
            loss_to_head, _ = model(t, r_, h)  # (t,r',h)
            # loss = loss.mean()
            loss = (loss_to_tail + loss_to_head)/2
            loss.backward()
            optimizer.step()
            full_loss.append(loss.item())
    return full_loss


def train_bais_ConvE_label(dataset, model, optimizer, device, batch_size=None):
    data = DataLoader(dataset.data['train'], batch_size, drop_last=False, shuffle=True)
    if model.multi_infer.pre_train_flag:
        model.multi_infer.pre_train_flag = 0
        ConvModel = model.multi_infer.ConvModel
        ConvModel.train()
        rel_cnt = model.relation_cnt
        optimizer = torch.optim.Adam(ConvModel.parameters(), lr=0.003)
        for _ in range(0, 120):
            full_loss = []
            for batch_data in tqdm(data):
                h = batch_data[0].to(device)
                t = batch_data[1].to(device)
                r = batch_data[2].to(device)
                r_ = batch_data[2].to(device) + rel_cnt
                optimizer.zero_grad()
                loss_to_tail, _, _ = ConvModel(h, r, t)  # (h,r,t)
                loss_to_head, _, _ = ConvModel(t, r_, h)  # (t,r',h)
                # loss = loss.mean()
                loss = (loss_to_tail + loss_to_head)/2
                loss.backward()
                optimizer.step()
                full_loss.append(loss.item())
            print(np.mean(full_loss))

    ConvModel = model.multi_infer.ConvModel
    rel_cnt = model.relation_cnt
    full_loss = []
    ConvModel.eval()
    for batch_data in tqdm(data):
        h = batch_data[0].to(device)
        t = batch_data[1].to(device)
        r = batch_data[2].to(device)
        r_ = batch_data[2].to(device) + rel_cnt
        loss_to_tail, _, _ = ConvModel(h, r, t)  # (h,r,t)
        loss_to_head, _, _ = ConvModel(t, r_, h)  # (t,r',h)
        # loss = loss.mean()
        loss = (loss_to_tail + loss_to_head)/2
        full_loss.append(loss.item())
    print(np.mean(full_loss))

    if not model.multi_infer.eval_ConvModel:
        #optimizer1 = torch.optim.Adam(model.multi_infer.ConvR.parameters(), lr=0.0001)
        optimizer2 = torch.optim.Adam(model.bais_infer.parameters(), lr=0.003)
        optimizer3 = torch.optim.Adam(model.multi_infer.attention.parameters(), lr=0.003)
        rel_cnt = model.relation_cnt
        full_loss = []
        for batch_data in tqdm(data):
            h = batch_data[0].to(device)
            t = batch_data[1].to(device)
            r = batch_data[2].to(device)
            r_ = batch_data[2].to(device) + rel_cnt
            #optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            loss_to_tail, _ = model(h, r, t)  # (h,r,t)
            loss_to_head, _ = model(t, r_, h)  # (t,r',h)
            # loss = loss.mean()
            loss = (loss_to_tail + loss_to_head)/2
            loss.backward()
            #optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            full_loss.append(loss.item())
    else:
        model.multi_infer.eval_ConvModel = 0
    return full_loss

def train_HF_NMlpE(data, model, optimizer, device):
    full_loss = []
    model.train()
    rel_cnt = model.relation_cnt
    for batch_data in tqdm(data):
        h = batch_data[0].to(device)
        t = batch_data[1].to(device)
        r = batch_data[2].to(device)
        r_ = batch_data[2].to(device) + rel_cnt
        optimizer.zero_grad()
        loss_to_tail, _ = model(h, r, t)  # (h,r,t)
        loss_to_head, _ = model(t, r_, h)  # (t,r',h)
        # loss = loss.mean()
        loss = (loss_to_tail + loss_to_head) / 2
        try:
            loss.backward()
            optimizer.step()
            full_loss.append(loss.item())
        except RuntimeError:
            full_loss.append(1)

    return full_loss

def train_triples_Transformer_MlpE(data, model, optimizer, device):
    full_loss = []
    model.train()
    rel_cnt = model.relation_cnt
    for batch_data in tqdm(data):
        h = batch_data[0].to(device)
        t = batch_data[1].to(device)
        r = batch_data[2].to(device)
        r_ = batch_data[2].to(device) + rel_cnt
        optimizer.zero_grad()
        loss_to_tail, _ = model(h, r, t)  # (h,r,t)
        loss_to_head, _ = model(t, r_, h)  # (t,r',h)
        # loss = loss.mean()
        loss = (loss_to_tail + loss_to_head) / 2
        loss.backward()
        optimizer.step()
        full_loss.append(loss.item())


    return full_loss
