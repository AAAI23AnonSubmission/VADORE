# -*- coding: utf-8 -*-
"""
Compute recall@100
"""


import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("rank_file")
# .csv file with columns uid, iid, rank,
# as output of eval_vad1.py for instance
args = parser.parse_args()

ranks = pd.read_csv(args.rank_file)
ranks['sub100'] = ranks['rank'] <= 100

print('R@100', ranks.groupby('uid')['sub100'].mean().mean())
