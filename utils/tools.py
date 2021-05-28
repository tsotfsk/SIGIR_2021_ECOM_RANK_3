import datetime
import os
import random

import models
import numpy as np
import pandas as pd
import torch

from utils.logger import Logger


def init_env(args):
    set_seed(args.seed)
    year, month, day = datetime.datetime.now(
    ).year, datetime.datetime.now().month, datetime.datetime.now().day
    if not os.path.exists('./log'):
        os.mkdir('./log')
    logger = Logger(f'./log/{args.model}_{year}_{month}_{day}.log')
    print_args(args, logger)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    return logger


def get_model(model_name):
    model_class = getattr(models, model_name)
    return model_class


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def dict2str(result_dict):
    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ':' + '%.04f' % value + '  '
    return result_str


def print_args(args, logger):
    for name, value in vars(args).items():
        if not name.startswith('item'):
            logger.info('%s=%s' % (name, value))


def uit_data(name):
    if os.path.exists(f'./dataset/{name}.csv'):
        return pd.read_csv(f'./dataset/{name}.csv')
    df = pd.read_csv(f'./{name}.csv')
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['timestamp'] = df['event_time'].astype(int) / 10**9
    df = df[['user_id', 'product_id', 'timestamp']]
    df.sort_values('timestamp', inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.to_csv(f'./dataset/{name}.csv')
#     df.columns = ['user_id:token', 'product_id:token', 'timestamp:float']
    return df


def load_data():
    train = uit_data('train')
    test = uit_data('test')

    train['phase'] = 'train'
    test['phase'] = 'test'
    data = pd.concat([train, test])

    # map
    new_ids, id_map = pd.factorize(data['product_id'])
    item_map = {i + 1: n for i, n in enumerate(id_map.tolist())}
    data['product_id'] = new_ids + 1

    train_df = data[data['phase'] == 'train']
    test_df = data[data['phase'] == 'test']
    return train_df, test_df, item_map


def save_csv(g, stage, phase):
    # sava test data
    with open(f'./dataset/{stage}/{phase}.seq', 'w') as f:
        for user_id, seq, target in zip(g.uids, g.seqs, g.tars):
            f.write(','.join([str(user_id), '|'.join(map(str, seq)), str(target)]))
            f.write('\n')
