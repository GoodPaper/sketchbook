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

from datetime import datetime
import plotly.graph_objects as go
import os, \
    pandas as pd, \
    numpy as np

ROOT = os.path.dirname( __file__ )
RESOURCE = os.path.join( ROOT, 'res' )


if __name__ == '__main__':

    # Dataset preparation
    crypto_df = pd.read_csv( os.path.join( RESOURCE, 'train.csv' ) )
    print( crypto_df.head( 10 ) )

    asset_details = pd.read_csv( os.path.join( RESOURCE, 'asset_details.csv' ) )
    print( asset_details )

    btc = crypto_df[ crypto_df[ 'Asset_ID' ] == 1 ].set_index( 'timestamp' )
    btc_mini = btc.iloc[ -200: ]

    fig = go.Figure( data = [
        go.Candlestick(
            x = btc_mini.index,
            open = btc_mini[ 'Open' ], high = btc_mini[ 'High' ],
            low = btc_mini[ 'Low' ], close = btc_mini[ 'Close' ]
        )
    ] )
    fig.show()

    # Preprocessing
    eth = crypto_df[ crypto_df[ 'Asset_ID' ] == 6 ].set_index( 'timestamp' )
    eth.info( show_counts = True )
    eth.isna().sum()

    btc.head()
    _frm = 'datetime64[s]'
    beg_btc = btc.index[ 0 ].astype( _frm )
    end_btc = btc.index[ -1 ].astype( _frm )
    beg_eth = eth.index[ 0 ].astype( _frm )
    end_eth = eth.index[ -1 ].astype( _frm )

    print( 'BTC data goes from ', beg_btc, 'to ', end_btc )
    print( 'Ethereum data goes from ', beg_eth, 'to ', end_eth )
    ( eth.index[ 1: ] - eth.index[ :-1 ] ).value_counts().head()

    # Data visualization



