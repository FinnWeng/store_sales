# from preprocess import Preprocess
import pandas as pd
from pandas.tseries.offsets import DateOffset
import csv
import numpy as np
import tensorflow as tf

from net.regression_net import Regression_Net



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


def pd_preprocess(train, test, stores, oil, transactions, steps):
    
    train.drop(['id'], axis='columns', inplace=True)
    test.drop(['id'], axis='columns', inplace=True)

    # print(train.head(30))
    # print(stores.head(30))

    # new_train =  pd.concat([train,stores], axis = 1)
    new_train =  pd.merge(train, stores,  how='left', left_on='store_nbr', right_on = 'store_nbr')

    sales = pd.DataFrame()
    copy_sales = pd.DataFrame()

    sales["sales"] = train["sales"]

    # print("train['sales'].isnull().values.any():", train["sales"].isnull().values.any())

    

    # print(new_train.head(30))
    # print(sales.head(30))

    new_test =  pd.merge(test, stores,  how='left', left_on='store_nbr', right_on = 'store_nbr')

    train_length = new_train.shape[0]



    data = new_train.append(new_test,sort=False, ignore_index=True)
    data['sales'] = data['sales'].fillna(0)

    # print(data.head(10))

    # print(new_train.info())
    # print(new_test.info())
    # print(data.info())

    data = pd.merge(data, oil,  how='left', left_on='date', right_on = 'date')

    data = pd.merge(data, transactions,  how='left', left_on=['date', "store_nbr"], right_on = ['date', "store_nbr"])
    




    data["dcoilwtico"] = data.dcoilwtico.fillna(data.dcoilwtico.mean())

    data["date"] = pd.to_datetime(data["date"])

    

    '''
    compute the previous day sales of specific store_nbr and family
    '''
    # group_data = pd.DataFrame()
    # group_data["date"] = data["date"] + DateOffset(days=1)
    # group_data["store_nbr"] = data["store_nbr"]
    # group_data["family"] = data["family"]
    # group_data["prev_family_sales"] = data["sales"].fillna(0)
    # group_data = group_data.groupby(["store_nbr","family"])

    # first_date = pd.DataFrame()
    # first_date["date"] = data["date"].query("date == '2013-01-01' ")


    # data.drop(['sales'], axis='columns', inplace=True)

    # data = pd.merge(data, group_data,  how='left', left_on=['date', "store_nbr", "family"], right_on = ['date', "store_nbr", "family"])

    # print(data.info())

    '''
    normalize onpromotion(test have), dcoilwtico(test have), transactions(test have)
    '''
    data[ 'onpromotion' ] = (data[ 'onpromotion' ])/ (data[ 'onpromotion' ].max())
    data[ 'dcoilwtico' ] = (data[ 'dcoilwtico' ])/ (data[ 'dcoilwtico' ].max())
    data[ 'transactions' ] = (data[ 'transactions' ])/ (data[ 'transactions' ].max())



    

    '''
    get dummy for family, type, date, 
    model should generalize through "store_nbr", 'family' and "city" and "state", 
    '''
    # family = pd.get_dummies( data['family'] , prefix = 'family' )
    # print(family.head(30))
    data["date"] = pd.to_datetime(data["date"])
    data["day_of_the_week"] = data["date"].dt.day_name()
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day

    make_dummy_list = ["day","day_of_the_week", "month", "cluster", "type", "city", "state" ]
    dummy_df_list = []
    for item_name in make_dummy_list:
        item_df = pd.get_dummies( data[item_name] , prefix = item_name )
        dummy_df_list.append(item_df)
        data.drop([item_name], axis='columns', inplace=True)
        
    # data.drop(["date"], axis='columns', inplace=True)

    dummy_df_list.append(data)
    data = pd.concat(dummy_df_list, axis=1)

    # data.drop(['sales'], axis='columns', inplace=True)


    '''
    Group up
    '''

    data = data.set_index(['store_nbr', 'family','date' ]).sort_index()
    data["transactions"]=data["transactions"].transform(lambda x: x.fillna(x.mean())) # fill by group mean


    print("data.isnull().values.any():", data.isnull().values.any())


    print(data.info())

    train_data = data.query("date < '2017-08-16'")
    test_data = data.query("date >= '2017-08-16'")

    print("train_data:",train_data.shape)
    print("test_data:", test_data.shape)

    # print(data.shape)

    # train_data = train_data.reset_index()
    # test_data = test_data.reset_index()

    # print("train_data:",train_data.shape)
    # print("test_data:", test_data.shape)

    train_group_size_array = train_data.reset_index().groupby(['store_nbr', 'family']).size().to_numpy()
    test_group_size_array = test_data.reset_index().groupby(['store_nbr', 'family']).size().to_numpy()


    print("group_size_array", np.max(train_group_size_array)) # 1684 days
    print("group_size_array", np.min(train_group_size_array)) # 1684 days
    print("group_size_array", np.max(test_group_size_array)) # 1684 days
    print("group_size_array", np.min(test_group_size_array)) # 1684 days


    train_y = train_data["sales"].to_numpy()
    print("train_y:",train_y.shape)


    train_data.drop(["sales"], axis='columns', inplace=True)
    test_data.drop(["sales"], axis='columns', inplace=True)

    train_x = train_data.to_numpy()
    test_x = test_data.to_numpy()

    # print("train_x:",train_x.shape)
    # print("test_x:",test_x.shape)

    train_x = np.reshape(train_x, [-1, 1684,train_x.shape[-1]])
    train_y = np.reshape(train_y, [-1, 1684,1])

    test_x = np.reshape(test_x, [-1, 16, test_x.shape[-1]])
    
    # print("train_x:")
    # print(train_x[0])



    last_train_x = train_x[:,-steps:,:]

    train_x = make_time_series_array(train_x, steps)


    test_x = np.concatenate([last_train_x, test_x], axis = 1)

    test_x = make_time_series_array(test_x, steps)
    train_y = train_y[:, steps+1:, :]

    train_x = np.reshape(train_x, [-1, train_x.shape[2],train_x.shape[3]])
    train_y = np.reshape(train_y, [-1,train_y.shape[2] ])

    print("train_x:",train_x.shape)
    print("test_x:",test_x.shape)
    print("train_y:",train_y.shape)

    test_x = np.reshape(test_x, [-1, test_x.shape[2], test_x.shape[3]])


    return train_x, train_y, test_x


def make_time_series_array(data_array, steps):
    '''
    steps not include itself. set to 7
    '''
    array_list = []
    length_of_array = data_array.shape[1]
    for step in range(steps+1):
        inner_array = data_array[:,step:step + (length_of_array - (steps+1)),:]
        array_list.append(inner_array)
        # print("inner_array:",inner_array.shape)
    
    time_series_array = np.stack(array_list,axis = 2)
    print(time_series_array.shape)
    return time_series_array



if __name__ == "__main__":
    '''
    each row of train.csv is a sale summation of a category(family) of a store of a day
    1. merge train/test with stores.csv by store_nub
    2. merge train/test with oil.csv by date
    3. merge transaction, with datatime and store_nbr
    4. holiday event, merge by date, then decide how to merge by type of holiday, then location and transferred
    '''

    pd.set_option('display.max_rows', None)

    config = define_config()
    model_config = define_model_config()

    time_steps = 7

    stores = pd.read_csv("./data/stores.csv")
    oil = pd.read_csv("./data/oil.csv")
    transactions = pd.read_csv("./data/transactions.csv")

    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    train_x, train_y, test_x = pd_preprocess(train, test, stores, oil, transactions, time_steps)

    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)

    ds_train_x = tf.data.Dataset.from_tensor_slices(train_x)
    ds_train_y = tf.data.Dataset.from_tensor_slices(train_y)
    ds_train = tf.data.Dataset.zip((ds_train_x, ds_train_y))
    ds_train = ds_train.map(ds_preprocess,  tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(config.shuffle_buffer)
    ds_train = ds_train.batch(config.batch_size, drop_remainder=False)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    
    







    
    


 

    


    