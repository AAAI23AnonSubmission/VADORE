"""
Computes and saves top-1000 recommendations on the training set according to the
Vadore.0 modules.
These top-1000 recommendations are later used for training the Vad.1 module
"""

import argparse
import pickle
import datetime as dt

import numpy as np
import pandas as pd
from sklearn import datasets
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler
from nn_utils import ReluTanhEncoder, BilinearForm

parser = argparse.ArgumentParser()
parser.add_argument("dropoutnet_dir")
# Should point to the recsys2017.pub/ file,
# obtained from the DropoutNet paper (Vokovs et al., 2017) github:
# https://github.com/layer6ai-labs/DropoutNet
parser.add_argument("model_dir")
# Directory created by the train_vad0 script, containing the Vad.0 models.
parser.add_argument("destfile")
# Directory in which to the recommendations (in a csv format)
args = parser.parse_args()


xing_path = '/data/rctnet/VADORE/CodeGuillaume/Xing/'

dropoutnet_feature_path = xing_path + 'recsys2017.pub/recsys2017.pub/eval/'
dropoutnet_path = xing_path + 'recsys2017.pub/recsys2017.pub/eval/warm/'

user_feature_path = args.dropoutnet_dir + 'eval/user_features_0based.txt'
item_feature_path = args.dropoutnet_dir + 'eval/item_features_0based.txt'

user_features, _ = datasets.load_svmlight_file(user_feature_path,
                                               zero_based=True, dtype=np.float32)
user_features = user_features.tolil().toarray()

item_features, _ = datasets.load_svmlight_file(item_feature_path,
                                               zero_based=True, dtype=np.float32)
item_features = item_features.tolil().toarray()

with open(args.model_dir + '/user_scaler.pkl', 'rb') as f:
    user_scaler = pickle.loads(f.read())

user_features = user_scaler.transform(user_features)
user_features = torch.Tensor(user_features)

with open(args.model_dir + '/item_scaler.pkl', 'rb') as f:
    item_scaler = pickle.loads(f.read())

item_features = item_scaler.transform(item_features)
item_features = torch.Tensor(item_features)


device = torch.device('cuda:0')

encoder_user = torch.load(args.model_dir + '/encoder_user.pt')
encoder_user.eval()
encoder_item = torch.load(args.model_dir + '/encoder_item.pt')
encoder_item.eval()
bilinear = torch.load(args.model_dir + '/bilinear.pt')
bilinear.eval()

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

train_users_path = args.dropoutnet_dir + 'eval/warm/train.csv'
train_users = pd.read_csv(train_users_path)
train_users.columns = ['uid', 'iid', 'inter', 'date']
train_users = train_users.loc[train_users.inter != 0]
train_users = train_users.drop_duplicates('uid')

preds = torch.Tensor(np.repeat(np.inf, len(item_features))).to(device)

with torch.no_grad():
    train_users_embedding = encoder_user.forward(user_features[train_users.uid.to_numpy()].to(device))
train_users_embedding = train_users_embedding.detach().cpu().numpy()

k = 1000
top = []
for chunk in tqdm(np.array_split(np.arange(len(train_users)),
                                 int(len(train_users) / 1000))):
    preds = np.matmul(train_users_embedding[chunk], final_embedding)
    top_idx = np.argpartition(preds, k, axis=1)[:, :k]
    top_values = np.take_along_axis(preds, top_idx, axis=1)
    top_ranks = np.argsort(top_values, axis=1)
    final_ranks = np.vstack([[top_idx[i, top_ranks[i]]] for i in range(len(chunk))])
    top += [final_ranks]

top = np.vstack(top)
top = top.squeeze()
top = pd.DataFrame(top, columns=['pred_' + str(k) for k in range(top.shape[1])])
top['id'] = train_users.uid.to_numpy()

top.to_csv(args.destfile, index=False)
