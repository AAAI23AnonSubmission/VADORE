"""
This script is concerned with the training of the so-called Vad.0 module,
which serves as a filter for Vad.1.
"""

import argparse
import pickle
import random
import os
import time

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.optim as optim
from nn_utils import ReluTanhEncoder, BilinearForm

parser = argparse.ArgumentParser()
parser.add_argument("dropoutnet_dir")
# Should point to the recsys2017.pub/ file,
# obtained from the DropoutNet paper (Vokovs et al., 2017) github:
# https://github.com/layer6ai-labs/DropoutNet
parser.add_argument("destdir")
# Directory in which to save models and losses
args = parser.parse_args()

start = time.perf_counter()

os.mkdir(args.destdir)

random.seed(3082022)
np.random.seed(3082022)

train = pd.read_csv(args.dropoutnet_dir + 'eval/warm/train.csv', header=None)
train.columns = ['uid', 'iid', 'inter', 'date']
train = train.loc[train.inter != 0] # Remove mere impressions
train = train.drop_duplicates(['uid', 'iid'])

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
pickle.dump(user_scaler, open(args.destdir + '/user_scaler.pkl', 'wb'))

item_scaler = StandardScaler()
item_features = item_scaler.fit_transform(item_features)
pickle.dump(item_scaler, open(args.destdir + '/item_scaler.pkl', 'wb'))

batch_size = 128
n_epochs = 50
learning_rate = 0.0001

device = torch.device('cuda:0')

user_features = torch.Tensor(user_features).to(device)
item_features = torch.Tensor(item_features)
print(user_features.shape, item_features.shape)

encoder_user = ReluTanhEncoder(user_features.shape[1], 800, 800, 400).to(device)
encoder_item = ReluTanhEncoder(item_features.shape[1], 800, 800, 400).to(device)
bilinear = BilinearForm(400, 400).to(device)
params = (list(encoder_user.parameters()) + list(encoder_item.parameters())
          + list(bilinear.parameters()))
optimizer = optim.Adam(params, lr=learning_rate)
criterion = torch.nn.MarginRankingLoss(margin=1.0)

train_uids = train.uid.to_numpy()
train_iids = train.iid.to_numpy()

losses = []
for _ in range(n_epochs):
    print(_)
    indices = np.arange(len(train))
    np.random.shuffle(indices)
    batches = np.array_split(indices, int(len(train) / batch_size))
    negatives = np.random.choice(item_features.shape[0], len(train))
    running_loss = 0.0
    for batch in tqdm(batches):
        optimizer.zero_grad()
        anchor_indices = train_uids[batch]
        pos_indices = train_iids[batch]
        neg_indices = negatives[batch]
        anchors_batch = user_features[anchor_indices]
        positives_batch = item_features[pos_indices].to(device)
        negatives_batch = item_features[neg_indices].to(device)
        anchors_batch = encoder_user.forward(anchors_batch)
        positives_batch = encoder_item.forward(positives_batch)
        negatives_batch = encoder_item.forward(negatives_batch)
        pos_scores = bilinear(anchors_batch, positives_batch)
        neg_scores = bilinear(anchors_batch, negatives_batch)
        loss = criterion(pos_scores, neg_scores,
                         -torch.ones(len(pos_scores)).reshape(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(running_loss)
    losses += [running_loss]
pd.DataFrame(losses).to_csv(args.destdir + '/losses.csv')

torch.save(encoder_user, args.destdir + "/encoder_user.pt")
torch.save(encoder_item, args.destdir + "/encoder_item.pt")
torch.save(bilinear, args.destdir + "/bilinear.pt")

stop = time.perf_counter()

print("Elapsed time (seconds):", stop - start)
