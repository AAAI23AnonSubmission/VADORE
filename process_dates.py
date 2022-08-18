# -*- coding: utf-8 -*-
"""
Preprocessing of the original items.csv in the RecSys 2017 Challenge data
(see http://www.recsyschallenge.com/2017/ for a description)
to obtain the item_dates.csv.
"""

import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("items_file")
# items.csv file in the original RecSys 2017 Challenge data
parser.add_argument("dest_file")
# .csv file to save item ids & dates to.
args = parser.parse_args()

items = pd.read_csv(args.items_file, engine='python', sep="\t")
items.columns = [x.split('.')[1] for x in items.columns]
items = items[['id', 'created_at']]
items.to_csv(args.dest_file)
