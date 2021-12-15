# from preprocess import Preprocess
import pandas as pd
import csv
import numpy as np
import tensorflow as tf

# from net.cls import Classification_Net



class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def define_config():
    config = AttrDict()
    config.shuffle_buffer = 2000
    config.batch_size = 32
    config.base_lr = 1e-3
    config.log_dir = "./tf_log/"
    config.model_path = './model/titanic_cls.ckpt'
    config.size_of_ds = 891
    config.steps_per_epoch = (config.size_of_ds//config.batch_size) * 10
    

    return config

def define_model_config():
    config = AttrDict()
    config.mlp_dim = 64
    config.layer_num = 8
    config.out_dim = 2
    return config


def ds_preprocess(train, test, stores, oil, transactions):
    
    train.drop(['id'], axis='columns', inplace=True)
    test.drop(['id'], axis='columns', inplace=True)

    # print(train.head(30))
    # print(stores.head(30))

    # new_train =  pd.concat([train,stores], axis = 1)
    new_train =  pd.merge(train, stores,  how='left', left_on='store_nbr', right_on = 'store_nbr')

    sales = pd.DataFrame()
    copy_sales = pd.DataFrame()

    sales["sales"] = train["sales"]

    new_train.drop(['sales'], axis='columns', inplace=True)

    # print(new_train.head(30))
    # print(sales.head(30))

    new_test =  pd.merge(test, stores,  how='left', left_on='store_nbr', right_on = 'store_nbr')

    train_length = new_train.shape[0]




    data = new_train.append(new_test,sort=False, ignore_index=True)

    # print(data.head(10))

    # print(new_train.info())
    # print(new_test.info())
    # print(data.info())

    data = pd.merge(data, oil,  how='left', left_on='date', right_on = 'date')

    data = pd.merge(data, transactions,  how='left', left_on=['date', "store_nbr"], right_on = ['date', "store_nbr"])

    
    data["dcoilwtico"] = data.dcoilwtico.fillna(data.dcoilwtico.mean())

    '''
    get dummy for store_nbr family, city, state, type, date, 
    '''
    family = pd.get_dummies( data['family'] , prefix = 'family' )
    # print(family.head(30))
    data["date"] = pd.to_datetime(data["date"])
    data["day_of_the_week"] = data["date"].dt.day_name()
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day

    make_dummy_list = ["day","day_of_the_week", "month", "store_nbr", "family","city", "state","cluster", "type" ]
    dummy_df_list = []
    for item_name in make_dummy_list:
        item_df = pd.get_dummies( data[item_name] , prefix = item_name )
        dummy_df_list.append(item_df)
        data.drop([item_name], axis='columns', inplace=True)
        
    data.drop(["date"], axis='columns', inplace=True)

    dummy_df_list.append(data)
    data = pd.concat(dummy_df_list, axis=1)

    # print(data.info())

    data = data.to_numpy(dtype=np.float32)
    sales = sales.to_numpy(dtype=np.float32)

    # print(data.shape)

    train_x = data[:train_length]
    test_x = data[train_length:]

    train_y = sales

    return train_x, train_y, test_x






if __name__ == "__main__":
    '''
    each row of train.csv is a sale summation of a category(family) of a store of a day
    1. merge train/test with stores.csv by store_nub
    2. merge train/test with oil.csv by date
    3. merge transaction, with datatime and store_nbr
    4. holiday event, merge by date, then decide how to merge by type of holiday, then location and transferred
    '''

    config = define_config()
    model_config = define_model_config()

    stores = pd.read_csv("./data/stores.csv")
    oil = pd.read_csv("./data/oil.csv")
    transactions = pd.read_csv("./data/transactions.csv")

    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    train_x, train_y, test_x = ds_preprocess(train, test, stores, oil, transactions)

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)







    
    


 

    


    