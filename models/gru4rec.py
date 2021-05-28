import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from models.layers import DynamicGRU, BPRLoss
import torch.nn.functional as F


class GRU4Rec(nn.Module):
    def __init__(self, config, n_items=None, device=None):
        super(GRU4Rec, self).__init__()

        # load parameters info
        self.config = config
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        # self.dropout_prob = config.dropout_prob
        self.n_items = n_items

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0)
        # self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = DynamicGRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.loss_type = "ce"
        if self.loss_type == "bpr":
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        if isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

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
        test_items_emb = self.item_embedding.weight[1:]
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores

    def save_model(self):
        state = {
            'config': self.config,
            'state_dict': self.state_dict(),
        }
        torch.save(state, './saved/GRU4Rec.pth')

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def __str__(self):
        return self.__class__.__name__
