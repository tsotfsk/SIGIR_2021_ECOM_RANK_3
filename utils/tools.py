import datetime
import json
import os
import pickle
import random
import time

import models
import numpy as np
import pandas as pd
import torch

import tqdm
from utils.logger import Logger
from dotenv import load_dotenv
from utils.uploader import upload_submission

# load envs from env file
load_dotenv(verbose=True, dotenv_path='./utils/upload.env.local')

EMAIL = os.getenv('EMAIL', None)  # the e-mail you used to sign up
assert EMAIL is not None


class EasyDict():
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def test_remap(uids, iids):
    with open('./dataset/raw/rec_test_phase_1.json') as json_file:
        # read the test cases from the provided file
        test_queries = json.load(json_file)

    with open('./dataset/new/map_info.pkl', 'rb') as f:
        # read the test cases from the provided file
        info = EasyDict(pickle.load(f))

    with open('./results/deepwalk_i_s_u.pkl', 'rb') as f:
        # read the test cases from the provided file
        dw_uids, dw_iids = pickle.load(f)

    uids = [info.idx2sess[uid] for uid in uids]
    iids = [[info.idx2item[iid] for iid in ilst] for ilst in iids]
    preds = dict(zip(uids, iids))

    dw_uids = [info.idx2sess[uid] for uid in dw_uids]
    dw_iids = [[info.idx2item[iid] for iid in ilst] for ilst in dw_iids]
    dw_preds = dict(zip(dw_uids, dw_iids))

    all_items = list(info.item2idx.keys())
    my_predictions = []
    missing = 0
    for t in tqdm.tqdm(test_queries, total=len(test_queries)):
        # this is our prediction, which defaults to a random SKU
        next_sku = np.random.choice(len(all_items), 20)
        next_sku = [info.idx2item[iid] for iid in next_sku]
        # copy the test case
        _pred = dict(t)

        session_id_hash = t['query'][0]['session_id_hash']
        if session_id_hash in preds:
            next_sku = preds[session_id_hash]
        elif session_id_hash in dw_preds:
            next_sku = dw_preds[session_id_hash]
        else:
            missing += 1

        # assert isinstance(next_sku, str)

        # append the label - which needs to be a list
        _pred["label"] = next_sku
        # append prediction to the final list
        my_predictions.append(_pred)

    print('缺失比例:{}'.format(missing / len(test_queries)))
    # name the prediction file according to the README specs
    local_prediction_file = '{}_{}.json'.format(
        EMAIL.replace('@', '_'), round(time.time() * 1000))

    # dump to file
    with open(local_prediction_file, 'w') as outfile:
        json.dump(my_predictions, outfile, indent=2)

    # finally, upload the test file using the provided script
    upload_submission(local_file=local_prediction_file, task='rec')
    # bye bye
    print("\nAll done at {}: see you, space cowboy!".format(
        datetime.datetime.utcnow()))


def load_data(path):
    with open(path + 'browsing.pkl', 'rb') as f:
        browsing = pickle.load(f)
    print('load browsing done...')
    with open(path + 'search.pkl', 'rb') as f:
        search = pickle.load(f)
    print('load search done...')
    with open(path + 'sku_to_content.pkl', 'rb') as f:
        sku = pickle.load(f)
    print('load sku done...')
    with open(path + 'map_info.pkl', 'rb') as f:
        info = pickle.load(f)
    print('load info done...')
    return browsing, search, sku, EasyDict(info)


def init_env(args):
    set_seed(args.seed)
    year, month, day = datetime.datetime.now(
    ).year, datetime.datetime.now().month, datetime.datetime.now().day
    if not os.path.exists('./log'):
        os.mkdir('./log')
    logger = Logger(f'./log/{args.model}_{year}_{month}_{day}.log')
    print_args(args, logger)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
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
            f.write(
                ','.join([str(user_id), '|'.join(map(str, seq)), str(target)]))
            f.write('\n')
