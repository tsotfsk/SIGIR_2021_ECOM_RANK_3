import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from models.layers import DynamicGRU, BPRLoss
import torch.nn.functional as F
import pickle
import numpy as np


class SUMLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, item_seq_emb, seq_len):
        item_seq_emb = item_seq_emb.sum(1)
        return item_seq_emb / seq_len.unsqueeze(1)


class AVERAGE(nn.Module):
    def __init__(self, config, n_items=None, device=None):
        super(AVERAGE, self).__init__()

        # load parameters info
        self.config = config
        self.embedding_size = 128
        self.hidden_size = 128
        self.num_layers = config.num_layers
        self.dropout_prob = 0.3
        self.n_items = n_items

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0)
        # self.img_embedding = nn.Embedding(
        #     self.n_items + 1, 50, padding_idx=0)
        self.txt_embedding = nn.Embedding(
            self.n_items + 1, 50, padding_idx=0)

        with open('./dataset/prepared/dw_sku.pkl', 'rb') as f:
            dw_item = pickle.load(f)
        with open('./dataset/prepared/text.pkl', 'rb') as f:
            text = pickle.load(f)
        # with open('./dataset/prepared/imag.pkl', 'rb') as f:
        #     imag = pickle.load(f)

        self.sum_layer = SUMLayer()

        dw_item = torch.from_numpy(dw_item)
        dw_item = F.normalize(dw_item, dim=1)

        text = torch.from_numpy(text)
        text = F.normalize(text, dim=1)

        # imag = torch.from_numpy(imag)
        # imag = F.normalize(imag, dim=1)

        # pretrain = torch.cat((text, imag), dim=1)
        with torch.no_grad():
            self.item_embedding.weight[1:].copy_(dw_item)
        #     self.img_embedding.weight[1:].copy_(imag)
            self.txt_embedding.weight[1:].copy_(text)
        self.item_embedding.weight.requires_grad = False
        # self.img_embedding.weight.requires_grad = False
        self.txt_embedding.weight.requires_grad = False
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # self.dense = nn.Linear(128, self.n_items + 1)
        self.loss_type = "ce"
        if self.loss_type == "bpr":
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # if isinstance(module, nn.Embedding):
        #     xavier_normal_(module.weight)
        if isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        # img_seq_emb = self.emb_dropout(self.img_embedding(item_seq))
        txt_seq_emb = self.txt_embedding(item_seq)
        # item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        ht_0 = self.sum_layer(item_seq_emb, item_seq_len)
        # ht_1 = self.sum_layer(img_seq_emb, item_seq_len)
        ht_1 = self.sum_layer(txt_seq_emb, item_seq_len)
        # ht = torch.cat(
        #     (ht_0, ht_1, ht_2), dim=1)
        test_items_emb = self.item_embedding.weight
        scores_0 = torch.matmul(
            ht_0, test_items_emb.transpose(0, 1))  # [B, n_items]

        test_txt_emb = self.txt_embedding.weight
        scores_1 = torch.matmul(
            ht_1, test_txt_emb.transpose(0, 1))  # [B, n_items]
        return scores_0 + 0.1 * scores_1

    def calculate_loss(self, feed_dict):
        item_seq = feed_dict['seqs']
        item_seq_len = feed_dict['seq_lens']
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = feed_dict['target_ids']
        loss = self.loss_fct(seq_output, pos_items)
        # torch.nn.Parameter(torch.zeros(1))
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, feed_dict):
        item_seq = feed_dict['seqs']
        item_seq_len = feed_dict['seq_lens']
        seq_output = self.forward(item_seq, item_seq_len)
        return seq_output

    def save_model(self):
        state = {
            'config': self.config,
            'state_dict': self.state_dict(),
        }
        torch.save(state, './saved/AVERAGE.pth')

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def __str__(self):
        return self.__class__.__name__
