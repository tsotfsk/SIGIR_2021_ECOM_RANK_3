import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from models.layers import DynamicGRU, BPRLoss
import torch.nn.functional as F
import pickle
import numpy as np


class GRU4Rec(nn.Module):
    def __init__(self, config, n_items=None, device=None):
        super(GRU4Rec, self).__init__()

        # load parameters info
        self.config = config
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.n_items = n_items
        self.normalize = config.normalize
        self.commit = config.commit

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0)

        # with open('./dataset/prepared/text.pkl', 'rb') as f:
        #     text = pickle.load(f)
        # with open('./dataset/prepared/imag.pkl', 'rb') as f:
        #     imag = pickle.load(f)

        # text = torch.from_numpy(text)
        # text = F.normalize(text, dim=1)

        # imag = torch.from_numpy(imag)
        # imag = F.normalize(imag, dim=1)

        # pretrain = torch.cat((text, imag), dim=1)
        # with torch.no_grad():
        #     self.item_embedding.weight[1:].copy_(pretrain)

        with open('./dataset/prepared/text.pkl', 'rb') as f:
            text = pickle.load(f)
        text = torch.from_numpy(text)
        text = F.normalize(text, dim=1)
        with torch.no_grad():
            self.item_embedding.weight[1:].copy_(text)

        # with open('./dataset/prepared/dw_sku_i-i.pkl', 'rb') as f:
        #     dw_item = pickle.load(f)
        # dw_item = torch.from_numpy(dw_item)
        # dw_item = F.normalize(dw_item, dim=1)
        # with torch.no_grad():
        #     self.item_embedding.weight[1:].copy_(dw_item)

        # self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = DynamicGRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        # self.dense = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.BatchNorm1d(self.hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(self.hidden_size, self.embedding_size),
        # )
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
            xavier_normal_(module.weight.data, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def load_embedding(self):
        with open('./dataset/prepared/dw_sku_i-i.pkl', 'rb') as f:
            item_embed = pickle.load(f)
        item_embed = torch.from_numpy(item_embed)
        dw_iitem_embedtem = F.normalize(dw_item, dim=1)
        with torch.no_grad():
            self.item_embedding.weight[1:].copy_(dw_iitem_embedtem)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        # item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        output, ht = self.gru_layers(item_seq_emb, item_seq_len)
        result = self.dense(ht.squeeze(0))
        return result

    def calculate_loss(self, feed_dict):
        item_seq = feed_dict['seqs']
        item_seq_len = feed_dict['seq_lens']
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = feed_dict['target_ids']

        if self.loss_type == "bpr":
            neg_items = feed_dict['neg_ids']
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, feed_dict):
        item_seq = feed_dict['seqs']
        item_seq_len = feed_dict['seq_lens']
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores

    def save_model(self):
        state = {
            'config': self.config,
            'state_dict': self.state_dict(),
        }
        torch.save(state, f'./saved/GRU4Rec_{self.commit}.pth')

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def __str__(self):
        return self.__class__.__name__
