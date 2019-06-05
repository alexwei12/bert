#!/usr/bin/env python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def train_valid_test_split(x_data, y_data,
        validation_size=0.1, test_size=0.1, shuffle=True):
    x_, x_test, y_, y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=shuffle)
    valid_size = validation_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(x_, y_, test_size=valid_size, shuffle=shuffle)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

if __name__ == '__main__':
    path = "D:\docs\TOON\内容与搜索\内容审核\数据\datagrand"
    pd_all = pd.read_csv(os.path.join(path, "t_datagrand_result_new.csv"))

    NONE_VIN = (pd_all["origin_value"].isnull()) | (pd_all["origin_value"].apply(lambda x: str(x).isspace()))| (pd_all["vltn_type"].apply(lambda x: str(x) == "Other"))
    pd_all = pd_all[~NONE_VIN]

    # pd_spam = pd.read_csv(os.path.join(path, "query_result_spam.csv"))
    # NONE_VIN = (pd_spam["origin_value"].isnull()) | (pd_spam["origin_value"].apply(lambda x: str(x).isspace()))
    # pd_spam = pd_spam[~NONE_VIN]

    # pd_all = shuffle(pd_all.sample(n=10000).append(pd_spam))
    x_data, y_data = pd_all.origin_value, pd_all.vltn_type

    x_train, x_valid, x_test, y_train, y_valid, y_test = \
            train_valid_test_split(x_data, y_data, 0.1, 0.01)

    data_dir="D:\project\\bert\data_all"

    train = pd.DataFrame({'label':y_train, 'x_train': x_train})
    train.to_csv(os.path.join(data_dir, "train.csv"), index=False, sep='\t')
    valid = pd.DataFrame({'label':y_valid, 'x_valid': x_valid})
    valid.to_csv(os.path.join(data_dir, "dev.csv"), index=False, sep='\t')
    test = pd.DataFrame({'label':y_test, 'x_test': x_test})
    test.to_csv(os.path.join(data_dir, "test.csv"), index=False, sep='\t')