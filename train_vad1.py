"""
This module trains the Vad.1 module, on top of the top-k values
for the training set.
"""

import argparse
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.optim as optim
from nn_utils import SecondStageRanker

random.seed(3082022)
np.random.seed(3082022)

parser = argparse.ArgumentParser()
parser.add_argument("dropoutnet_dir")
# Should point to the recsys2017.pub/ file,
# obtained from the DropoutNet paper (Vokovs et al., 2017) github:
# https://github.com/layer6ai-labs/DropoutNet
parser.add_argument("topfile")
# A .csv file containing top-1k matches for the train uid's
# (as produced by the select_top script)
parser.add_argument("destdir")
# Directory in which to save models and losses
args = parser.parse_args()

start = time.perf_counter()

os.mkdir(args.destdir)

train = pd.read_csv(args.dropoutnet_dir + 'eval/warm/train.csv', header=None)
train.columns = ['uid', 'iid', 'inter', 'date']
train = train.loc[train.inter != 0] # Discard impressions
train = train.drop_duplicates(['uid', 'iid'])

top = pd.read_csv(args.topfile)

top = top.melt('id')
top["rank"] = top.variable.str.replace('pred_', '').astype(int)
# Standardise ranks before providing them as inputs to the neural network
# (note that since the dataset contains ranks 0, 999) for all uids,
# the mean and standard deviation are known).
top['rank'] = (top['rank'] - 499.5) / 288.674
top = top.rename(columns={"value": "iid", "id": "uid"})
top = top[['uid', 'iid', 'rank']]

# In the training set, only retain those (uid, iid) pairs such that
# the iid is among the uid's top 1k job ads according to Vad.0.
train = train.merge(top, on=['uid', 'iid'], how='inner')

# The "top" DataFrame is now repurposed for sampling the "negative" iids
# i.e. job ads with which the uid did not match but are nevertheless
# in Vad.0's top 1k recommendations.
top = top.loc[top.uid.isin(train.uid)]
test_pairs = set((i, j) for i, j in zip(train.uid, train.iid))
top['ismatch'] = [(i, j) in test_pairs for i, j in zip(top.uid, top.iid)]
top = top.loc[top.ismatch == False]
n_to_sample = train.groupby('uid')['iid'].count()
n_to_sample.rename('n_sample', inplace=True)
top = top.set_index('uid').join(n_to_sample)

train = train.sort_values('uid')

top = top.sort_values('uid')

user_feature_path = args.dropoutnet_dir + 'eval/user_features_0based.txt'
item_feature_path = args.dropoutnet_dir + 'eval/item_features_0based.txt'

user_features, _ = datasets.load_svmlight_file(user_feature_path,
                                               zero_based=True, dtype=np.float32)
user_features = user_features.tolil().toarray()

item_features, _ = datasets.load_svmlight_file(item_feature_path,
                                               zero_based=True, dtype=np.float32)
item_features = item_features.tolil().toarray()

user_scaler = StandardScaler()
user_features = user_scaler.fit_transform(user_features)
pickle.dump(user_scaler, open('user_scaler.pkl', 'wb'))

item_scaler = StandardScaler()
item_features = item_scaler.fit_transform(item_features)
pickle.dump(item_scaler, open('item_scaler.pkl', 'wb'))


batch_size = 128
n_epochs = 25
learning_rate = 0.001

device = torch.device('cuda:0')

user_features = torch.Tensor(user_features).to(device)
item_features = torch.Tensor(item_features)
print(user_features.shape, item_features.shape)

model = SecondStageRanker(user_features.shape[1], item_features.shape[1] + 1,
                          200)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MarginRankingLoss(margin=1.0)

train_uids = train.uid.to_numpy()
train_iids = train.iid.to_numpy()

train_ranks = torch.Tensor(train['rank'].to_numpy()).to(device)

criterion = torch.nn.BCEWithLogitsLoss(reduce=False)

losses = []
for ep in range(n_epochs):
    print(ep)
    indices = np.arange(len(train))
    np.random.shuffle(indices)
    batches = np.array_split(indices, int(len(train) / batch_size))
    negs = top.groupby('uid').apply(lambda g: g[['iid', 'rank']].sample(g.n_sample.iloc[0]))
    negatives = negs.iid.to_numpy()
    negative_ranks = torch.Tensor(negs['rank'].to_numpy()).to(device)
    running_loss = 0.0
    for batch in tqdm(batches):
        optimizer.zero_grad()
        anchor_indices = train_uids[batch]
        pos_indices = train_iids[batch]
        neg_indices = negatives[batch]
        anchors_batch = user_features[anchor_indices]
        positives_batch = torch.hstack([item_features[pos_indices].to(device),
                                        train_ranks[batch].reshape(-1, 1).to(device)])
        negatives_batch = torch.hstack([item_features[neg_indices].to(device),
                                        negative_ranks[batch].reshape(-1, 1).to(device)])
        pos_scores = model.forward(anchors_batch, positives_batch)
        neg_scores = model.forward(anchors_batch, negatives_batch)

        ones = torch.ones(len(pos_scores)).reshape(-1, 1).to(device)
        zeros = torch.zeros(len(neg_scores)).reshape(-1, 1).to(device)
        loss = 0.0
        loss += (criterion(pos_scores, ones)).sum()
        loss += (criterion(neg_scores, zeros)).sum()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(running_loss)
    losses += [running_loss]
pd.DataFrame(losses).to_csv(args.destdir + '/losses.csv')

torch.save(model, args.destdir + "/model.pt")

stop = time.perf_counter()

print("Elapsed time (seconds):", stop - start)
