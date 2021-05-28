import numpy as np
import torch
import random
import pickle
import tqdm


class SeqLoader(object):

    def __init__(self, phase='train', device=None, neg_sample=-1, batch_size=1024) -> None:
        super().__init__()
        self.device = device
        self.neg_sample = neg_sample
        self.phase = phase

        self.batch_size = batch_size
        self.pr = 0

        self.n_items = 32784
        self.all_items = set(range(self.n_items))
        # data
        self.data_list = []
        self._load_data()

    @property
    def batch_num(self):
        return len(self.data_list) // self.batch_size

    def _load_data(self):
        with open('./dataset/new/map_info.pkl', 'rb') as f:
            info = pickle.load(f)

        self.n_items = len(info['idx2item'])
        with open(f'./dataset/prepared/{self.phase}.csv', 'r') as f:
            for line in tqdm.tqdm(f):
                user_id, seq, target_id = line.split(',')
                seq = list(map(int, seq.split('|')))
                seq_len = sum([1 for i in seq if i > 0])
                self.data_list.append(
                    (int(user_id), seq, seq_len, int(target_id)))

    def _neg_sample(self, batch_data):
        batch_data['neg_ids'] = []
        for seq in batch_data['seqs']:
            neg_ids = []
            seq_set = set(seq)
            while len(neg_ids) < self.neg_sample:
                neg_id = np.random.choice(self.n_items, 1)[0]
                neg_id += 1  # XXX pad at 0
                if neg_id in seq_set:
                    continue
                neg_ids.append(neg_id)
            batch_data['neg_ids'].append(neg_ids)
        return batch_data

    def _sample(self, seq, target_id):
        sample = []
        cand_set = list(self.all_items - set(seq) - {target_id})
        items = np.random.choice(cand_set, self.test_sample, replace=False)
        sample.extend((items + 1).tolist())
        return sample

    def shuffle(self):
        random.shuffle(self.data_list)

    def to(self, batch_data):
        for key, value in batch_data.items():
            batch_data[key] = torch.LongTensor(value).to(self.device)
        return batch_data

    def _list2dict(self, batch_data):
        result = {}
        user_ids, seqs, seq_lens, target_ids = list(
            map(list, zip(*batch_data)))
        result['user_ids'] = user_ids
        result['seqs'] = seqs
        result['seq_lens'] = seq_lens
        result['target_ids'] = target_ids
        return result

    def _prepare_one_batch(self):
        batch_data = self.data_list[self.pr:self.pr + self.batch_size]
        batch_data = self._list2dict(batch_data)
#         if self.training and self.neg_sample > 0:
#             batch_data = self._neg_sample(batch_data)
        self.pr += self.batch_size
        return self.to(batch_data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pr > len(self.data_list):
            self.pr = 0
            raise StopIteration()
        return self._prepare_one_batch()
