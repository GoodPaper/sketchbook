# %% [markdown]
# **Author:** hozilab
#
# **Description:**
#  본 스크립트는 암호화폐의 단기 가격을 예측하는 스크립트이다. 데이터는 G-Research Crypto Forecasting challenge의 데이터를 사용한다. 알고리즘은 RNN, LSTM을 사용할 것이며, **Raoul Malm**의 스크립트에 영감을 받아서 작성하였다. Thanks **Raoul Malm**.
#
# **Outline:**
# 1. [환경설정](#1-bullet)
# 2. [데이터 분석](#2-bullet)
# 3. [데이터 전처리](#3-bullet)
# 4. [모델 수립](#4-bullet)
# 5. [모델 검증](#5-bullet)
# 6. [예측](#6-bullet)
#
# **Reference:**
# * [NY Stock Price Prediction RNN LSTM GRU](https://www.kaggle.com/raoulma/ny-stock-price-prediction-rnn-lstm-gru/)
# * [G-Research Crypto Forecasting ](https://www.kaggle.com/c/g-research-crypto-forecasting)

# %% [markdown]
# # 1. 환경 설정 <a class="anchor" id="1-bullet"></a>

# %% [code] {"execution":{"iopub.status.busy":"2021-12-08T13:43:48.691815Z","iopub.execute_input":"2021-12-08T13:43:48.692214Z","iopub.status.idle":"2021-12-08T13:43:48.714216Z","shell.execute_reply.started":"2021-12-08T13:43:48.692102Z","shell.execute_reply":"2021-12-08T13:43:48.713113Z"}}
# # Load 3rd-party library
# # https://stackoverflow.com/questions/13128647/matplotlib-finance-volume-overlay
# import sys
# !{sys.executable} -m pip install mpl_finance

# %% [code] {"execution":{"iopub.status.busy":"2021-12-08T13:43:48.715739Z","iopub.execute_input":"2021-12-08T13:43:48.716223Z","iopub.status.idle":"2021-12-08T13:43:54.371906Z","shell.execute_reply.started":"2021-12-08T13:43:48.716187Z","shell.execute_reply":"2021-12-08T13:43:54.37103Z"}}
# Load default packages
from datetime import datetime
from collections import namedtuple
import matplotlib.pyplot as plt
import os, gc, sklearn, \
    numpy as np, \
    pandas as pd, \
    tensorflow as tf


# %% [code] {"execution":{"iopub.status.busy":"2021-12-08T13:43:54.373294Z","iopub.execute_input":"2021-12-08T13:43:54.373601Z","iopub.status.idle":"2021-12-08T13:44:48.303298Z","shell.execute_reply.started":"2021-12-08T13:43:54.373561Z","shell.execute_reply":"2021-12-08T13:44:48.302365Z"}}
# _keys = [
#     'BinanceCoin', 'Bitcoin', 'BitcoinCash', 'Cardano', 'Dogecoin',
#     'EOS_IO', 'Ethereum', 'EthereumClassic', 'IOTA', 'Litecoin',
#     'Maker', 'Monero', 'Stellar', 'TRON'
# ]
# _clrs = [
#     'red', 'sienna', 'coral', 'gold', 'yellow',
#     'lawngreen', 'green', 'lightseagreen', 'deepskyblue', 'blue',
#     'blueviolet', 'deeppink', 'gray', 'black'
# ]
_keys = ['BinanceCoin', 'Bitcoin']
_clrs = ['red', 'sienna']
_mold = namedtuple('DataSet', _keys)
_root = 'C:/Users/user/Desktop/gitrepo/sketchbook/ml/kaggle/g_research_crypto_forecasting/res'

# Key Map
KMAP = [(a, b) for a, b in zip(_keys, _clrs)]
print(KMAP)

# # Load details
# DATASET_DETAILS = pd.read_csv( os.path.join( _root, 'asset_details.csv' ) ) \
#     .sort_values( [ 'Asset_ID' ] ) \
#     .reset_index( drop = True )
# print( DATASET_DETAILS )
# print( 'Load details.\n\n' )

# Load train set
_d = pd.read_csv(os.path.join(_root, 'train.csv'))
DATASET_TRAINS = _mold(*[
    _d.query('Asset_ID == {}'.format(i)).set_index('timestamp')
    for i in range(0, 2)  # 14
])
# for x in DATASET_TRAINS:
#     print( x.head() )
print('Load trains...\n\n')

# # Load supplement dataset... Not Now
# _d = pd.read_csv( os.path.join( _root, 'supplemental_train.csv' ) )
# DATASET_SUPPLEMENTS = _mold( *[
#     _d.query( 'Asset_ID == {}'.format( i ) ).set_index( 'timestamp' )
#     for i in range( 0, 14 )
# ] )
# # for x in DATASET_SUPPLEMENTS:
# #     print( x.head() )
# print( 'Load supplements...\n\n' )

del _d
gc.collect()


# %% [markdown]
# # 2. 데이터 분석 <a class="anchor" id="2-bullet"></a>

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-12T05:13:36.420797Z","iopub.execute_input":"2021-12-12T05:13:36.421116Z","iopub.status.idle":"2021-12-12T05:13:42.900283Z","shell.execute_reply.started":"2021-12-12T05:13:36.421073Z","shell.execute_reply":"2021-12-12T05:13:42.899496Z"}}
# Header looks like...
# timestamp Asset_ID 	Count 	Open 	High 	Low 	Close 	Volume 	VWAP 	Target

def _drawSubPlot(indx):
    _what, _color = KMAP[indx]

    plt.subplot(15, 2, indx * 2 + 1)
    plt.plot(getattr(DATASET_TRAINS, _what).Close.values, color=_color, label='close')
    plt.title(_what)
    plt.xlabel('Minutes')
    plt.ylabel('price')
    plt.legend(loc='best')

    plt.subplot(15, 2, indx * 2 + 2)
    plt.plot(getattr(DATASET_TRAINS, _what).Volume.values, color=_color, label='volume')
    plt.title(_what)
    plt.xlabel('Minutes')
    plt.ylabel('volume')
    plt.legend(loc='best')


plt.figure(figsize=(15, 50))

for i in range(2):  # todo 14
    _drawSubPlot(i)

plt.tight_layout()
plt.show()
gc.collect()

# %% [markdown]
# # 3. 데이터 분석 <a class="anchor" id="3-bullet"></a>
# ## 3.1 Missing data 확인. 빈 값은 pad 형태로 채워넣기.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-12T05:13:42.902723Z","iopub.execute_input":"2021-12-12T05:13:42.903048Z","iopub.status.idle":"2021-12-12T05:13:42.949964Z","shell.execute_reply.started":"2021-12-12T05:13:42.903005Z","shell.execute_reply":"2021-12-12T05:13:42.949186Z"}}
for item in KMAP:
    name = item[0]
    what = getattr(DATASET_TRAINS, name)
    tcnt = len(what)
    print(name)
    report = (what.index[1:] - what.index[: -1]).value_counts().sort_index()
    _cnt60 = report[60]
    _cntes = sum(report[x] for x in report.index[1:])  # 60초가 아닌 것.
    print('{} = {} + {} [ {} ]\n'.format(tcnt, _cnt60, _cntes, tcnt - (_cnt60 + _cntes) - 1))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-12T05:13:42.951140Z","iopub.execute_input":"2021-12-12T05:13:42.951488Z","iopub.status.idle":"2021-12-12T05:13:43.761515Z","shell.execute_reply.started":"2021-12-12T05:13:42.951459Z","shell.execute_reply":"2021-12-12T05:13:43.760626Z"}}
# 이전 값으로 채우는 형태로 만들자.

wrap = dict()
for item in KMAP:
    name = item[0]
    what = getattr(DATASET_TRAINS, name)
    what = what.reindex(range(what.index[0], what.index[-1] + 60, 60), method='pad')  # 빈 데이터가 있을 수 있기에 Gap으로 채움.

    # TODO 현재 메모리 문제로 인해서 각 코인의 35%의 최신 데이터만 사용하는 것으로 변경
    bndl = int(len(what) * 0.65)
    wrap[name] = what[bndl:]

DATASET_TRAINS = _mold(**wrap)
del wrap
gc.collect()

for item in KMAP:
    name = item[0]
    what = getattr(DATASET_TRAINS, name)
    tcnt = len(what)
    print(name)
    report = (what.index[1:] - what.index[: -1]).value_counts().sort_index()
    _cnt60 = report[60]
    _cntes = sum(report[x] for x in report.index[1:])  # 60초가 아닌 것.
    print('{} = {} + {} [ {} ]\n'.format(tcnt, _cnt60, _cntes, tcnt - (_cnt60 + _cntes) - 1))

# %% [markdown]
# ## 3.2 각 데이터의 Normalize 수행( Scaling )

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-12T05:13:43.764375Z","iopub.execute_input":"2021-12-12T05:13:43.764890Z","iopub.status.idle":"2021-12-12T05:13:44.976748Z","shell.execute_reply.started":"2021-12-12T05:13:43.764845Z","shell.execute_reply":"2021-12-12T05:13:44.975803Z"}}
import sklearn.preprocessing

MIN_MAX_SCALER = sklearn.preprocessing.MinMaxScaler()


def normalize(df):
    # timestamp Asset_ID 	Count 	Open 	High 	Low 	Close 	Volume 	VWAP 	Target
    df['Open'] = MIN_MAX_SCALER.fit_transform(df.Open.values.reshape(-1, 1))
    df['High'] = MIN_MAX_SCALER.fit_transform(df.High.values.reshape(-1, 1))
    df['Low'] = MIN_MAX_SCALER.fit_transform(df.Low.values.reshape(-1, 1))
    df['Close'] = MIN_MAX_SCALER.fit_transform(df.Close.values.reshape(-1, 1))
    df['Volume'] = MIN_MAX_SCALER.fit_transform(df.Volume.values.reshape(-1, 1))
    df['VWAP'] = MIN_MAX_SCALER.fit_transform(df.VWAP.values.reshape(-1, 1))
    return df


def prepareset(what, divide_at):
    data_raw = what.values  # convert to numpy array

    # create all possible sequences of length seq_len
    data = []
    for index in range(len(data_raw) - divide_at):
        data.append(data_raw[index: index + divide_at])
    data = np.array(data);

    valid_set_size = int(np.round(10 / 100 * data.shape[0]))
    test_set_size = int(np.round(10 / 100 * data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)

    x_train = data[: train_set_size, :-1, :]
    y_train = data[: train_set_size, -1, :]

    x_valid = data[train_set_size: train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size: train_set_size + valid_set_size, -1, :]

    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]


# choose one coin, Remove unnecessary features.
pick_coin = DATASET_TRAINS.Bitcoin.copy()
pick_coin.drop(['Asset_ID', 'Target'], 1, inplace=True)
print(pick_coin.columns)

# normalize stock
# pick_coin_norm = pick_coin.copy()
pick_coin = normalize(pick_coin)

# create train, test data
seq_len = 20
x_train, y_train, x_valid, y_valid, x_test, y_test = prepareset(pick_coin, seq_len)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ', x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T05:13:44.978050Z","iopub.execute_input":"2021-12-12T05:13:44.978290Z","iopub.status.idle":"2021-12-12T05:13:47.956492Z","shell.execute_reply.started":"2021-12-12T05:13:44.978261Z","shell.execute_reply":"2021-12-12T05:13:47.955605Z"}}
# Normalize 한 데이터 어떻게 보여지는지 확인

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(pick_coin.Open.values, color='red', label='open')
plt.plot(pick_coin.Close.values, color='green', label='low')
plt.plot(pick_coin.Low.values, color='blue', label='low')
plt.plot(pick_coin.High.values, color='black', label='high')

plt.title('bitcoin')
plt.xlabel('Minutes')
plt.ylabel('normalized price/volume')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(pick_coin.Volume.values, color='gray', label='volume')
plt.title('Volume')
plt.xlabel('Minutes')
plt.ylabel('volume')
plt.legend(loc='best')

plt.show()

# %% [markdown]
# # 4. 모델 만들기 <a class="anchor" id="4-bullet"></a>
# ## 4.1 RNN, LSTM, GRU로 돌려보자.

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T05:33:43.370487Z","iopub.execute_input":"2021-12-12T05:33:43.371312Z","iopub.status.idle":"2021-12-12T05:34:34.549406Z","shell.execute_reply.started":"2021-12-12T05:33:43.371265Z","shell.execute_reply":"2021-12-12T05:34:34.548017Z"}}
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

## Basic Cell RNN in tensorflow

index_in_epoch = 0
perm_array = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)


# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)  # shuffle permutation array
        start = 0  # start next epoch
        index_in_epoch = batch_size

    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]


# parameters
n_steps = seq_len - 1
n_inputs = 7
n_neurons = 200
n_outputs = 7
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 100
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# 기본 RNN Cell로 돌려 보기
# layers = [
#     tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
#     for layer in range(n_layers)
# ]

# use LSTM Cell with peephole connections
layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=n_neurons, activation=tf.nn.leaky_relu, use_peepholes = True)
         for layer in range(n_layers)]

# use GRU cell
# layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#          for layer in range(n_layers)]

multi_layer_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:, n_steps - 1, :]  # keep only last output of sequence

loss = tf.reduce_mean(tf.square(outputs - y))  # loss function = mean squared error
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

from datetime import datetime
TSMP = lambda: datetime.now().strftime( '%Y%m%d-%H%M%S' )

# run graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loop = int(n_epochs * train_set_size / batch_size)
    for iteration in range( loop ):
        x_batch, y_batch = get_next_batch(batch_size)  # fetch the next training batch
        #         print( x_batch.shape, y_batch.shape )
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
        if iteration % int(5 * train_set_size / batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid})
            print( TSMP() + ' [%d - %d]%.2f epochs: MSE train/valid = %.6f/%.6f' % (
                iteration, loop,
                iteration * batch_size / train_set_size, mse_train, mse_valid
            ) )

    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})



ft = 4 # Count 	Open 	High 	Low 	Close 	Volume 	VWAP

plt.figure(figsize=(15, 5));
plt.subplot(1, 2, 1);

plt.plot(np.arange(y_train.shape[0]), y_train[:, ft], color='blue', label='train target')

plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_valid.shape[0]), y_valid[:, ft],
         color='gray', label='valid target')

plt.plot(np.arange(y_train.shape[0] + y_valid.shape[0],
                   y_train.shape[0] + y_test.shape[0] + y_test.shape[0]),
         y_test[:, ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0]), y_train_pred[:, ft], color='red',
         label='train prediction')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0] + y_valid_pred.shape[0]),
         y_valid_pred[:, ft], color='orange', label='valid prediction')

plt.plot(np.arange(y_train_pred.shape[0] + y_valid_pred.shape[0],
                   y_train_pred.shape[0] + y_valid_pred.shape[0] + y_test_pred.shape[0]),
         y_test_pred[:, ft], color='green', label='test prediction')

plt.title('past and future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');

plt.subplot(1, 2, 2);

plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]),
         y_test[:, ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0] + y_test_pred.shape[0]),
         y_test_pred[:, ft], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');

corr_price_development_train = np.sum(np.equal(np.sign(y_train[:, 1] - y_train[:, 0]),
                                               np.sign(y_train_pred[:, 1] - y_train_pred[:, 0])).astype(int)) / \
                               y_train.shape[0]
corr_price_development_valid = np.sum(np.equal(np.sign(y_valid[:, 1] - y_valid[:, 0]),
                                               np.sign(y_valid_pred[:, 1] - y_valid_pred[:, 0])).astype(int)) / \
                               y_valid.shape[0]
corr_price_development_test = np.sum(np.equal(np.sign(y_test[:, 1] - y_test[:, 0]),
                                              np.sign(y_test_pred[:, 1] - y_test_pred[:, 0])).astype(int)) / \
                              y_test.shape[0]

print('correct sign prediction for close - open price for train/valid/test: %.2f/%.2f/%.2f' % (
    corr_price_development_train, corr_price_development_valid, corr_price_development_test))