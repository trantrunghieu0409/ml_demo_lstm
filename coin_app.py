import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import requests
from datetime import datetime
import pandas as pd

app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

exchange = 'Coinbase'
datetime_interval = 'day'

def get_filename(from_symbol, to_symbol, exchange, datetime_interval, download_date):
    return '%s_%s_%s_%s_%s.csv' % (from_symbol, to_symbol, exchange, datetime_interval, download_date)

def download_data(from_symbol, to_symbol, exchange, datetime_interval):
    supported_intervals = {'minute', 'hour', 'day'}
    assert datetime_interval in supported_intervals,\
        'datetime_interval should be one of %s' % supported_intervals
    print('Downloading %s trading data for %s %s from %s' %
          (datetime_interval, from_symbol, to_symbol, exchange))
    base_url = 'https://min-api.cryptocompare.com/data/histo'
    url = '%s%s' % (base_url, datetime_interval)
    params = {'fsym': from_symbol, 'tsym': to_symbol,
              'limit': 200, 'aggregate': 1,
              'e': exchange}
    request = requests.get(url, params=params)
    data = request.json()
    return data

def convert_to_dataframe(data):
    df = pd.json_normalize(data, ['Data'])
    df['datetime'] = pd.to_datetime(df.time, unit='s')
    df = df[['datetime', 'low', 'high', 'open',
             'close', 'volumefrom', 'volumeto']]
    return df

def filter_empty_datapoints(df):
    indices = df[df.sum(axis=1) == 0].index
    print('Filtering %d empty datapoints' % indices.shape[0])
    df = df.drop(indices)
    return df

def predict(from_symbol, to_symbol, model):
    data = download_data(from_symbol, to_symbol, exchange, datetime_interval)
    df_nse = convert_to_dataframe(data)

    test_size = 0.3
    split_row = len(df_nse) - int(test_size * len(df_nse))

    df_nse["datetime"]=pd.to_datetime(df_nse.datetime,format="%Y-%m-%d")
    df_nse.index=df_nse['datetime']

    data=df_nse.sort_index(ascending=True,axis=0)
    new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['datetime','close'])

    for i in range(0,len(data)):
        new_data["datetime"][i]=data['datetime'][i]
        new_data["close"][i]=data["close"][i]

    new_data.index=new_data.datetime
    new_data.drop("datetime",axis=1,inplace=True)

    dataset=new_data.values

    train=dataset[0:split_row,:]
    valid=dataset[split_row:,:]

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)

    x_train,y_train=[],[]

    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
        
    x_train,y_train=np.array(x_train),np.array(y_train)

    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    model=load_model(model)

    inputs=new_data[len(new_data)-len(valid)-60:].values
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)

    X_test=[]
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test=np.array(X_test)

    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    closing_price=model.predict(X_test)
    closing_price=scaler.inverse_transform(closing_price)

    train=new_data[:split_row]
    valid=new_data[split_row:]
    valid['predictions']=closing_price
    return train, valid

train_BTC, valid_BTC = predict(from_symbol='BTC', to_symbol='USD', model='saved_BTC_model.h5')
train_ADA, valid_ADA = predict(from_symbol='ADA', to_symbol='USD', model='saved_ADA_model.h5')
train_ETH, valid_ETH = predict(from_symbol='ETH', to_symbol='USD', model='saved_ETH_model.h5')

def visualize(name, coin, train, valid):
    return dcc.Tab(label=name,children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data " + coin,
					figure={
						"data":[
							go.Scatter(
								x=train.index,
								y=valid["close"],
								mode='lines'
							)

						],
						"layout":go.Layout(
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data " + coin,
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=valid["predictions"],
								mode='lines'
							)

						],
						"layout":go.Layout(
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])        		
        ])

app.layout = html.Div([
   
    html.H1("Digital Coin Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
        visualize(name="BTC-USD Coin Data",train=train_BTC,valid=valid_BTC, coin='BTC'),
        visualize(name="ADA-USD Coin Data",train=train_ADA,valid=valid_ADA, coin='ADA'),
        visualize(name="ETH-USD Coin Data",train=train_ETH,valid=valid_ETH, coin='ETH'),
    ])
])

if __name__=='__main__':
	app.run_server(debug=True)