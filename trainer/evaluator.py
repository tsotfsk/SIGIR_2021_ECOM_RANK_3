import torch
import numpy as np


class FullEvaluator(object):

    def __init__(self, metrics=None, topk=10, pos_len=1):
        self.topk = topk
        self.metrics = metrics
        self.pos_len = pos_len

    def collect(self, scores, batch_data):
        # intermediate variables
        target_ids = batch_data['target_ids'].cpu().numpy()
        # idx = (torch.argmax(scores, dim=-1)).cpu().numpy()  # nusers x k

        idx = torch.topk(scores, k=self.topk, dim=1)[1].cpu().numpy()

        return ((target_ids[:, np.newaxis] - idx) == 0).sum(1), idx

    def evaluate(self, matrix):
        pos_idx = np.concatenate(matrix, axis=0)
        return pos_idx.sum() / pos_idx.shape[0]
