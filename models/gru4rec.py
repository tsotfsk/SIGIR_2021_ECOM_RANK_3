import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_

from models.layers import DynamicGRU


class GRU4Rec(nn.Module):
    def __init__(self, config, n_items=None, device=None):
        super(GRU4Rec, self).__init__()

        # load parameters info
        self.config = config
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_items = n_items
        self.normalize = config.normalize
        self.commit = config.commit

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0)

        if self.commit != 'rand':
            item_embed = self.load_embedding()
            dw_iitem_embedtem = F.normalize(item_embed, dim=1)
            with torch.no_grad():
                self.item_embedding.weight[1:].copy_(dw_iitem_embedtem)

        self.gru_layers = DynamicGRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
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

    @property
    def commit2embed(self):
        return {
            'txt': './dataset/prepared/text.pkl',
            'dw': './dataset/prepared/dw_sku.pkl',
            'dw_i-i': './dataset/prepared/dw_sku_i-i.pkl'
        }

    def load_embedding(self):
        with open(self.commit2embed[self.commit], 'rb') as f:
            item_embed = pickle.load(f)
        return torch.from_numpy(item_embed)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        _, ht = self.gru_layers(item_seq_emb, item_seq_len)
        result = self.dense(ht.squeeze(0))
        return result

    def calculate_loss(self, feed_dict):
        item_seq = feed_dict['seqs']
        item_seq_len = feed_dict['seq_lens']
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = feed_dict['target_ids']

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
