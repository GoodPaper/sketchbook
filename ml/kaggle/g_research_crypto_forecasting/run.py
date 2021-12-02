'''
Over $40 billion worth of cryptocurrencies are traded every day. They are among the most popular assets for speculation and investment, yet have proven wildly volatile. Fast-fluctuating prices have made millionaires of a lucky few, and delivered crushing losses to others. Could some of these price movements have been predicted in advance?

In this competition, you'll use your machine learning expertise to forecast short term returns in 14 popular cryptocurrencies. We have amassed a dataset of millions of rows of high-frequency market data dating back to 2018 which you can use to build your model. Once the submission deadline has passed, your final score will be calculated over the following 3 months using live crypto data as it is collected.

The simultaneous activity of thousands of traders ensures that most signals will be transitory, persistent alpha will be exceptionally difficult to find, and the danger of overfitting will be considerable. In addition, since 2018, interest in the cryptomarket has exploded, so the volatility and correlation structure in our data are likely to be highly non-stationary. The successful contestant will pay careful attention to these considerations, and in the process gain valuable insight into the art and science of financial forecasting.

G-Research is Europe’s leading quantitative finance research firm. We have long explored the extent of market prediction possibilities, making use of machine learning, big data, and some of the most advanced technology available. Specializing in data science and AI education for workforces, Cambridge Spark is partnering with G-Research for this competition. Watch our introduction to the competition below:

This is a Code Competition



Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

    CPU Notebook <= 9 hours run-time
    GPU Notebook <= 9 hours run-time
    Internet access disabled
    Freely & publicly available external data is allowed, including pre-trained models
    Submission file must be named submission.csv

Please see the Code Competition FAQ for more information on how to submit. Review the code debugging doc if you are encountering submission errors.

Submissions are evaluated on a weighted version of the Pearson correlation coefficient. You can find additional details in the 'Prediction Details and Evaluation' section of this tutorial notebook.

You must submit to this competition using the provided python time-series API, which ensures that models do not peek forward in time. To use the API, follow this template in Kaggle Notebooks:

import gresearch_crypto
env = gresearch_crypto.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test set and sample submission
for (test_df, sample_prediction_df) in iter_test:
    sample_prediction_df['Target'] = 0  # make your predictions here
    env.predict(sample_prediction_df)   # register your predictions



https://www.kaggle.com/sohier/detailed-api-introduction

'''

'''
Condition
    CPU Notebook <= 9 hours run-time
    GPU Notebook <= 9 hours run-time
    Internet access disabled
    Freely & publicly available external data is allowed, including pre-trained models
    Submission file must be named submission.csv
    Using the provided python time-series API
'''


from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import scipy.stats as stats
import os, time, sys, \
    pandas as pd, \
    numpy as np


ROOT = os.path.dirname( __file__ )
RESOURCE = os.path.join( ROOT, 'res' )


if __name__ == '__main__':

    # ===================================== #
    # 1. 학습 데이터, 데이터 목록을 불러온다.
    # ===================================== #
    crypto_df = pd.read_csv( os.path.join( RESOURCE, 'train.csv' ) )
    asset_details = pd.read_csv( os.path.join( RESOURCE, 'asset_details.csv' ) ).sort_values( [ 'Asset_ID' ] )
    print( crypto_df.head( 10 ) )
    print( asset_details )

    # ===================================== #
    # 2. 비트코인, 이더리움 데이터를 가져온다. 타임스탬프를 인덱스로 설정한다.
    # ===================================== #
    btc = crypto_df[ crypto_df[ 'Asset_ID' ] == 1 ].set_index( 'timestamp' )
    eth = crypto_df[ crypto_df[ 'Asset_ID' ] == 6 ].set_index( 'timestamp' )

    # 2-1. 데이터 일부분을 그래프로 그려본다.
    btc_mini = btc.iloc[ -200: ]
    fig = go.Figure( data = [
        go.Candlestick(
            x = btc_mini.index,
            open = btc_mini[ 'Open' ], high = btc_mini[ 'High' ],
            low = btc_mini[ 'Low' ], close = btc_mini[ 'Close' ]
        )
    ] )
    fig.show()

    # ===================================== #
    # 3. 데이터를 점검한다.
    # ===================================== #
    eth.info( show_counts = True )
    '''
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1956200 entries, 1514764860 to 1632182400
    Data columns (total 9 columns):
     #   Column    Non-Null Count    Dtype  
    ---  ------    --------------    -----  
     0   Asset_ID  1956200 non-null  int64  
     1   Count     1956200 non-null  float64
     2   Open      1956200 non-null  float64
     3   High      1956200 non-null  float64
     4   Low       1956200 non-null  float64
     5   Close     1956200 non-null  float64
     6   Volume    1956200 non-null  float64
     7   VWAP      1956200 non-null  float64
     8   Target    1955860 non-null  float64
    dtypes: float64(8), int64(1)
    memory usage: 149.2 MB
    '''

    print( eth.isna().sum() )
    '''
    Asset_ID      0
    Count         0
    Open          0
    High          0
    Low           0
    Close         0
    Volume        0
    VWAP          0
    Target      340     Target 컬럼에 NA 값이 340개 있다고 한다. 데이터가 빠져있는 것임.
    dtype: int64
    '''

    print( btc.head() )
    '''
    timestamp Asset_ID 	Count 	Open 	High 	Low 	Close 	Volume 	VWAP 	Target
    1514764860 	1 	229.0 	13835.194 	14013.8 	13666.11 	13850.176 	31.550062 	13827.062093 	-0.014643
    1514764920 	1 	235.0 	13835.036 	14052.3 	13680.00 	13828.102 	31.046432 	13840.362591 	-0.015037
    1514764980 	1 	528.0 	13823.900 	14000.4 	13601.00 	13801.314 	55.061820 	13806.068014 	-0.010309
    1514765040 	1 	435.0 	13802.512 	13999.0 	13576.28 	13768.040 	38.780529 	13783.598101 	-0.008999
    1514765100 	1 	742.0 	13766.000 	13955.9 	13554.44 	13724.914 	108.501637 	13735.586842 	-0.008079    
    '''

    _frm = 'datetime64[s]'
    print(
        'BTC data goes from ', btc.index[ 0 ].astype( _frm ),
        'to ', btc.index[ -1 ].astype( _frm )
    )
    print(
        'Ethereum data goes from ', eth.index[ 0 ].astype( _frm ),
        'to ', eth.index[ -1 ].astype( _frm )
    )
    '''
    BTC data goes from  2018-01-01T00:01:00 to  2021-09-21T00:00:00
    Ethereum data goes from  2018-01-01T00:01:00 to  2021-09-21T00:00:00
    '''

    # Missing data가 NaN으로 채워지지 않고, Row가 누락된 형태로 되어 있다.
    # 각 Row 별 Timestamp의 차이를 확인해서 누락된 데이터가 어떻게 형성되어 있는지 확인해보자.
    print(
        ( eth.index[ 1: ] - eth.index[ :-1 ] ).value_counts()
            .sort_index()
            .head()
    )
    '''
    60      1956043         <- 기본 Interval임. 1분 단위
    120         100         <- 이 줄 포함해서 아래는 전부 Missing row라고 보면 됨. Gap 이 있다는 의미이기 때문이다. 참고로 이더리움 말고 비트코인도 누락된 부분 많음.
    180          23
    240           3
    300           5
    360           2
    420           1
    480           2
    540           4
    600           2
    780           1
    900           1
    960           1
    1020          1
    1080          1
    1320          1
    1380          1
    1860          1
    1980          1
    2220          1
    2580          1
    3120          1
    4740          1
    7200          1
    Name: timestamp, dtype: int64
    '''

    # 시계열에 빈 공간이 있는 것을 알았으니, 이를 처리해보자. 여기서는 값을 채울건데, 이전의 값을 이용해서 채우는 것( gap )으로 진행하겠다.
    # 참고로, 교수님께서도 ffill 이나 보간법을 사용하지 말고, 이전의 값을 그대로 사용하는 것이 좋다라고 말씀해 주셨었다.
    eth = eth.reindex(
        range( eth.index[ 0 ], eth.index[ -1 ] + 60, 60 ), # 처음부터, 마지막까지, 60초 단위로...
        method = 'pad' # 이전 값을 이용해 채우는 방법( GAP )을 이용
    )
    btc = btc.reindex(
        range( btc.index[ 0 ], btc.index[ -1 ] + 60, 60 ),
        method = 'pad'
    )
    # 그리고 재 확인.
    print(
        ( eth.index[ 1: ] - eth.index[ :-1 ] ).value_counts()
            .sort_index()
            .head()
    )
    '''
    60    1956959                       <- 전부 60초 Interval로 채워졌다. eth 데이터 개수가 1956960이니, Interval은 총 1956960 - 1 = 1956959가 맞음.
    Name: timestamp, dtype: int64
    '''

    # ===================================== #
    # 4. 눈으로 데이터 확인해보기.
    # ===================================== #
    # 비트코인, 이더리움의 종가를 확인해보자.

    # Plot 객체를 우선 만들어 놓고...
    f = plt.figure( figsize = ( 15, 4 ) )

    # 비트코인 그리기
    ax = f.add_subplot( 121 )
    plt.plot( btc[ 'Close' ], label = 'BTC' )
    plt.legend()
    plt.xlabel( 'Time' )
    plt.ylabel( 'Bitcoin' )

    # 이더리움 그리기
    ax2 = f.add_subplot( 122 )
    ax2.plot( eth[ 'Close' ], color = 'red', label = 'ETH' )
    plt.legend()
    plt.xlabel( 'Time' )
    plt.ylabel( 'Ethereum' )

    # 보여줘
    plt.tight_layout()
    plt.show()

    # 2021-01-06 ~ 2021-01-07 이틀간의 비트코인, 이더리움의 가격을 따로 떼어내서 살펴보자.
    # ( 상대적으로 ) 짧은 구간에서의 2개 자산을 뽑은 이유는 작은 Window 내에서 두 개의 자산간의 Correlation을 구하기 위함이다.
    totimestamp = lambda x: int( datetime.strptime( x, '%d/%m/%Y' ).timestamp() )
    _ts_20210106 = totimestamp( '01/06/2021' )
    _ts_20210107 = totimestamp( '01/07/2021' )
    btc_mini_2021 = btc.loc[ _ts_20210106 : _ts_20210107 ]
    eth_mini_2021 = eth.loc[ _ts_20210106 : _ts_20210107 ]

    # 이번에는 이틀간 데이터로 떼어낸 2개 자산의 종가에 대해서 시각화해서 살펴보자.
    f = plt.figure( figsize = ( 7, 8 ) )

    # BTC 확인
    ax = f.add_subplot( 211 )
    plt.plot( btc_mini_2021[ 'Close' ], label = 'btc' )
    plt.legend()
    plt.xlabel( 'Time' )
    plt.ylabel( 'Bitcoin Close' )

    # ETH 확인
    ax2 = f.add_subplot( 212 )
    ax2.plot( eth_mini_2021[ 'Close' ], color = 'red', label = 'eth' )
    plt.legend()
    plt.xlabel( 'Time' )
    plt.ylabel( 'Ethereum Close' )

    # 보여줘
    plt.tight_layout()
    plt.show()

    # TODO Log returns
    # define function to compute log returns
    def log_return(series, periods=1):
        return np.log(series).diff(periods=periods)

    lret_btc = log_return(btc_mini_2021.Close)[1:]
    lret_eth = log_return(eth_mini_2021.Close)[1:]
    lret_btc.rename('lret_btc', inplace=True)
    lret_eth.rename('lret_eth', inplace=True)

    plt.figure(figsize=(8, 4))
    plt.plot(lret_btc)
    plt.plot(lret_eth)
    plt.show()

    # TODO Correlation between assets
    # join two asset in single DataFrame

    lret_btc_long = log_return(btc.Close)[1:]
    lret_eth_long = log_return(eth.Close)[1:]
    lret_btc_long.rename('lret_btc', inplace=True)
    lret_eth_long.rename('lret_eth', inplace=True)
    two_assets = pd.concat([lret_btc_long, lret_eth_long], axis=1)

    # group consecutive rows and use .corr() for correlation between columns
    corr_time = two_assets.groupby(two_assets.index // (10000 * 60)).corr().loc[:, "lret_btc"].loc[:, "lret_eth"]

    corr_time.plot()
    plt.xticks([])
    plt.ylabel("Correlation")
    plt.title("Correlation between BTC and ETH over time")

    all_assets_2021 = pd.DataFrame([])
    for asset_id, asset_name in zip(asset_details.Asset_ID, asset_details.Asset_Name):
        asset = crypto_df[crypto_df["Asset_ID"] == asset_id].set_index("timestamp")
        asset = asset.loc[totimestamp('01/01/2021'):totimestamp('01/05/2021')]
        asset = asset.reindex(range(asset.index[0], asset.index[-1] + 60, 60), method='pad')
        lret = log_return(asset.Close.fillna(0))[1:]
        all_assets_2021 = all_assets_2021.join(lret, rsuffix=asset_name, how="outer")

    plt.imshow(all_assets_2021.corr())
    plt.yticks(asset_details.Asset_ID.values, asset_details.Asset_Name.values)
    plt.xticks(asset_details.Asset_ID.values, asset_details.Asset_Name.values, rotation='vertical')
    plt.colorbar()

    # TODO Building our prediction model
    # TODO  Prediction targets and evaluation
    # TODO  Feature design
    # Select some input features from the trading data:
    # 5 min log return, abs(5 min log return), upper shadow, and lower shadow.
    upper_shadow = lambda asset: asset.High - np.maximum(asset.Close, asset.Open)
    lower_shadow = lambda asset: np.minimum(asset.Close, asset.Open) - asset.Low

    X_btc = pd.concat([log_return(btc.VWAP, periods=5), log_return(btc.VWAP, periods=1).abs(),
                       upper_shadow(btc), lower_shadow(btc)], axis=1)
    y_btc = btc.Target

    X_eth = pd.concat([log_return(eth.VWAP, periods=5), log_return(eth.VWAP, periods=1).abs(),
                       upper_shadow(eth), lower_shadow(eth)], axis=1)
    y_eth = eth.Target

    # TODO  Preparing the data for building predictive models
    # select training and test periods
    train_window = [totimestamp("01/05/2021"), totimestamp("30/05/2021")]
    test_window = [totimestamp("01/06/2021"), totimestamp("30/06/2021")]

    # divide data into train and test, compute X and y
    # we aim to build simple regression models using a window_size of 1
    X_btc_train = X_btc.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()  # filling NaN's with zeros
    y_btc_train = y_btc.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()

    X_btc_test = X_btc.loc[test_window[0]:test_window[1]].fillna(0).to_numpy()
    y_btc_test = y_btc.loc[test_window[0]:test_window[1]].fillna(0).to_numpy()

    X_eth_train = X_eth.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()
    y_eth_train = y_eth.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()

    X_eth_test = X_eth.loc[test_window[0]:test_window[1]].fillna(0).to_numpy()
    y_eth_test = y_eth.loc[test_window[0]:test_window[1]].fillna(0).to_numpy()

    # simple preprocessing of the data
    scaler = StandardScaler()

    X_btc_train_scaled = scaler.fit_transform(X_btc_train)
    X_btc_test_scaled = scaler.transform(X_btc_test)

    X_eth_train_scaled = scaler.fit_transform(X_eth_train)
    X_eth_test_scaled = scaler.transform(X_eth_test)

    # TODO  Baseline model: Linear regression
    # implement basic ML baseline (one per asset)
    lr = LinearRegression()
    lr.fit(X_btc_train_scaled, y_btc_train)
    y_pred_lr_btc = lr.predict(X_btc_test_scaled)

    lr.fit(X_eth_train_scaled, y_eth_train)
    y_pred_lr_eth = lr.predict(X_eth_test_scaled)

    # we concatenate X and y for both assets
    X_both_train = np.concatenate((X_btc_train_scaled, X_eth_train_scaled), axis=1)
    X_both_test = np.concatenate((X_btc_test_scaled, X_eth_test_scaled), axis=1)
    y_both_train = np.column_stack((y_btc_train, y_eth_train))
    y_both_test = np.column_stack((y_btc_test, y_eth_test))

    # define the direct multioutput model and fit it
    mlr = MultiOutputRegressor(LinearRegression())
    lr.fit(X_both_train, y_both_train)
    y_pred_lr_both = lr.predict(X_both_test)

    # TODO  Evaluate baselines
    print('Test score for LR baseline: BTC', f"{np.corrcoef(y_pred_lr_btc, y_btc_test)[0,1]:.2f}",
          ', ETH', f"{np.corrcoef(y_pred_lr_eth, y_eth_test)[0,1]:.2f}")
    print('Test score for multiple output LR baseline: BTC', f"{np.corrcoef(y_pred_lr_both[:,0], y_btc_test)[0,1]:.2f}",
          ', ETH', f"{np.corrcoef(y_pred_lr_both[:,1], y_eth_test)[0,1]:.2f}")