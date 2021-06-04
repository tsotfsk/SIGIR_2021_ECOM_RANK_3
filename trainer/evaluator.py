import torch
import numpy as np


class FullEvaluator(object):

    def __init__(self, metrics=None, topk=20, pos_len=1):
        self.topk = topk
        self.metrics = metrics
        self.pos_len = pos_len

    def collect(self, scores, batch_data):
        # intermediate variables
        target_ids = batch_data['target_ids'].cpu()
        # idx = (torch.argmax(scores, dim=-1)).cpu().numpy()  # nusers x k

        idx = torch.topk(scores, k=self.topk, dim=1)[1].cpu()

        return ((idx - target_ids.unsqueeze(1)) == 0), idx

    def evaluate(self, matrix):
        pos_idx = torch.cat(matrix, axis=0).float()
        ranks = 1 / (pos_idx.argmax(dim=1) + 1)
        mask = pos_idx.sum(1)
        result = ranks * mask
        return result.sum() / pos_idx.shape[0]
