# -*- coding: utf-8 -*-
"""
Evaluation script for the Vad.1 module.
"""

import argparse
import pickle
import datetime as dt

import numpy as np
import pandas as pd
from sklearn import datasets
from tqdm import tqdm
import torch
from scipy.stats import rankdata

from nn_utils import ReluTanhEncoder, BilinearForm

parser = argparse.ArgumentParser()
parser.add_argument("dropoutnet_dir")
# Should point to the recsys2017.pub/ file,
# obtained from the DropoutNet paper (Vokovs et al., 2017) github:
# https://github.com/layer6ai-labs/DropoutNet
parser.add_argument("vad0dir")
# A directory containing scalers and models for vad0
parser.add_argument("vad1dir")
# A directory containing the model.pt file for vad1
parser.add_argument("datefile")
# File containing item dates;
# corresponds to the dates.csv file provided in the present repository
# (see process_dates.py for its extraction from items.csv in the
# now unavailable 2017 original challenge data).
parser.add_argument('mode')
# String, either "warm" or "cold_user"
parser.add_argument("destfile")
# A .csv file in which to save computed rankings
args = parser.parse_args()

assert args.mode in ['warm', 'cold_user']

user_feature_path = args.dropoutnet_dir + 'eval/user_features_0based.txt'
item_feature_path = args.dropoutnet_dir + 'eval/item_features_0based.txt'

user_features, _ = datasets.load_svmlight_file(user_feature_path,
                                               zero_based=True, dtype=np.float32)
user_features = user_features.tolil().toarray()

item_features, _ = datasets.load_svmlight_file(item_feature_path,
                                               zero_based=True, dtype=np.float32)
item_features = item_features.tolil().toarray()

with open(args.vad0dir + '/user_scaler.pkl', 'rb') as f:
    user_scaler = pickle.loads(f.read())

user_features = user_scaler.transform(user_features)
user_features = torch.Tensor(user_features)

with open(args.vad0dir + '/item_scaler.pkl', 'rb') as f:
    item_scaler = pickle.loads(f.read())

item_features = item_scaler.transform(item_features)
item_features = torch.Tensor(item_features)

device = torch.device('cuda:0')

encoder_user = torch.load(args.vad0dir + '/encoder_user.pt')
encoder_user.eval()
encoder_item = torch.load(args.vad0dir + '/encoder_item.pt')
encoder_item.eval()
bilinear = torch.load(args.vad0dir + '/bilinear.pt')
bilinear.eval()

second_stage_model = torch.load(args.vad1dir + '/model.pt')
second_stage_model.eval()
second_stage_model = second_stage_model.cpu()

A = list(bilinear.parameters())[0].detach().cpu().numpy().squeeze()

encoder_user = encoder_user.to(device)
encoder_item = encoder_item.to(device)
bilinear = bilinear.to(device)

with torch.no_grad():
    final_embedding = [encoder_item.forward(item_features[z].float().to(device))
                       for z in np.array_split(np.arange(item_features.shape[0]),
                                               int(item_features.shape[0] / 10000))]

final_embedding = np.vstack([x.cpu().numpy() for x in final_embedding])
final_embedding = np.matmul(A, final_embedding.T)

if args.mode == 'cold_user':
    test_users_path = args.dropoutnet_dir + 'eval/warm/test_cold_user.csv'
else:
    test_users_path = args.dropoutnet_dir + 'eval/warm/test_warm.csv'

test_users = pd.read_csv(test_users_path)
test_users.columns = ['uid', 'iid', 'inter', 'date']
test_users['datetime'] = pd.to_datetime(test_users['date'], unit='s')

items = pd.read_csv(args.datefile)
items['datetime_offer'] = pd.to_datetime(items.created_at, unit='s')

min_start_date = test_users.datetime.min()
invalid_items = ((items['datetime_offer'] < min_start_date - dt.timedelta(days=60))
                 .to_numpy().reshape(-1, 1))
invalid_items = np.where(invalid_items)[0] + 1


preds = torch.Tensor(np.repeat(np.inf, len(items))).to(device)

test = test_users.groupby('uid')['iid'].apply(list)

test_users_tensor = user_features[test.index.to_numpy()]
with torch.no_grad():
    test_users_embedding = encoder_user.forward(test_users_tensor.to(device))
test_users_embedding = test_users_embedding.detach().cpu().numpy()

k = 1000
all_ranks = []
for chunk in tqdm(np.array_split(np.arange(len(test)),
                                 int(len(test) / 500))):
    preds = np.matmul(test_users_embedding[chunk], final_embedding)

    ranks = np.zeros(preds.shape)
    ranks[:] = np.inf

    top_idx = np.argpartition(preds, k, axis=1)[:, :k]

    top_values = np.take_along_axis(preds, top_idx, axis=1)

    top_ranks = np.argsort(top_values, axis=1)

    for i in range(len(chunk)):
        user_repr = test_users_tensor[chunk[i], :].reshape(1, -1).repeat_interleave(k, 0)
        item_repr = torch.hstack([item_features[top_idx[i]],
                                  torch.Tensor((top_ranks[i] - 499.5) / 288.674).reshape(-1, 1)])
        with torch.no_grad():
            second_stage_preds = second_stage_model.forward(user_repr, item_repr).flatten()
        second_stage_preds = -second_stage_preds.cpu().detach().numpy()
        second_stage_preds[np.isin(top_idx[i], invalid_items)] = np.inf
        ranks_second_stage = rankdata(second_stage_preds)
        ranks[i, top_idx[i]] = ranks_second_stage

    all_ranks += [ranks[i][truth] for i, truth in enumerate(test.iloc[chunk])]
    print(np.mean(np.array([np.mean(np.array(x) <= 100) for x in all_ranks])))

test = test.reset_index()
test = test.explode('iid')
test["rank"] = [i for sublist in all_ranks for i in sublist]
test.to_csv(args.destfile)
