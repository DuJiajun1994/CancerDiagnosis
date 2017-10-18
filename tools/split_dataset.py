# coding=utf-8

import pandas as pd
import numpy as np

train_df = pd.DataFrame(columns=['x1', 'x2', 'y1', 'y2', 'label'], dtype=int)
test_df = pd.DataFrame(columns=['x1', 'x2', 'y1', 'y2', 'label'], dtype=int)

annotation = '/home/dujiajun/thyroid nodule/annotation/position&size&TIRADS/benign tumour.csv'

df = pd.read_csv(annotation,
                 dtype={
                     'image_name': np.str,
                     'x1': np.int,
                     'x2': np.int,
                     'y1': np.int,
                     'y2': np.int,
                     'cancer_size': np.int,
                     'TIRADS': np.str
                 },
                 index_col='image_name')
grouped = df.groupby(['cancer_size', 'TIRADS'])

test_list = []
train_list = []
for key, val in grouped.groups.items():
    l = val.tolist()
    np.random.shuffle(l)
    test_size = len(l) / 7
    test_list += l[:test_size]
    train_list += l[test_size:]


for ind in test_list:
    test_df.loc[ind] = df.loc[ind]
    test_df.loc[ind]['label'] = 0

for ind in train_list:
    train_df.loc[ind] = df.loc[ind]
    train_df.loc[ind]['label'] = 0

test_df = test_df.sort_index().astype(np.int)
train_df = train_df.sort_index().astype(np.int)

train_annotation = '/home/dujiajun/thyroid nodule/annotation/train.csv'
train_df.to_csv(train_annotation)

test_annotation = '/home/dujiajun/thyroid nodule/annotation/test.csv'
test_df.to_csv(test_annotation)




