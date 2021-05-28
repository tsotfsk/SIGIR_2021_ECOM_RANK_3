import pandas as pd
import pickle


class EasyDict():
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


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
