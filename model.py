# 必要なライブラリのインポート
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import math
import numpy as np
import tensorflow as tf



# データの取得（USD/JPYの場合）
symbol = 'JPY=X'
df = yf.download(symbol, start='2015-01-01', end='2024-07-31')

# データの前処理と特徴量エンジニアリング
df.reset_index(inplace=True)
df = df[['Date', 'Close']]
df.columns = ['ds', 'y']

# データの正規化
scaler = MinMaxScaler(feature_range=(0, 1))
df['y'] = scaler.fit_transform(np.array(df['y']).reshape(-1, 1))

# データをトレーニング、検証、テストセットに分割
train_size = int(len(df) * 0.6)
val_size = int(len(df) * 0.2)
test_size = len(df) - train_size - val_size
train, val, test = df.iloc[0:train_size], df.iloc[train_size:train_size+val_size], df.iloc[train_size+val_size:]

# LSTMに適した形式にデータを変換する関数
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# データセットの作成
look_back = 30  # 過去のデータから予測する期間
trainX, trainY = create_dataset(train['y'].values.reshape(-1, 1), look_back)
valX, valY = create_dataset(val['y'].values.reshape(-1, 1), look_back)
testX, testY = create_dataset(test['y'].values.reshape(-1, 1), look_back)

# データのreshape（LSTMに入力する形式にする）
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
valX = np.reshape(valX, (valX.shape[0], valX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# LSTMモデルの作成
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# モデルのコンパイル
model.compile(optimizer='adam', loss='mean_squared_error')

# モデルのトレーニング
model.fit(trainX, trainY, epochs=80, batch_size=45, validation_data=(valX, valY))


 # モデルの保存
model.save('lstm_usdjpy_model.h5')

# 予測の実行
trainPredict = model.predict(trainX)
valPredict = model.predict(valX)
testPredict = model.predict(testX)

# 予測データの逆正規化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
valPredict = scaler.inverse_transform(valPredict)
valY = scaler.inverse_transform(valY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY.reshape(-1, 1))

# 精度の評価
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
valScore = math.sqrt(mean_squared_error(valY, valPredict))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print(f'Train Score: {trainScore:.2f} RMSE')
print(f'Validation Score: {valScore:.2f} RMSE')
print(f'Test Score: {testScore:.2f} RMSE')

# 未来の予測（7日分）
future_predictions = []
last_input = testX[-1]  # テストデータの最後の入力

for _ in range(30):
    prediction = model.predict(last_input.reshape(1, look_back, 1))
    future_predictions.append(prediction[0][0])
    last_input = np.append(last_input[1:], prediction, axis=0)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 未来予測のための日付の生成
last_date = df['ds'].values[-1]
future_dates = pd.date_range(start=last_date, periods=31, inclusive='right')

# プロット用のデータの準備
trainPredictPlot = np.empty_like(df['y'])
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict.flatten()

valPredictPlot = np.empty_like(df['y'])
valPredictPlot[:] = np.nan
valPredictPlot[len(trainPredict) + (look_back * 2):len(trainPredict) + (look_back * 2) + len(valPredict)] = valPredict.flatten()

testPredictPlot = np.empty_like(df['y'])
testPredictPlot[:] = np.nan
testPredictPlot[len(df) - len(testPredict):] = testPredict.flatten()

futurePredictPlot = np.empty(len(future_dates))
futurePredictPlot[:] = np.nan
futurePredictPlot[:len(future_predictions)] = future_predictions.flatten()

# グラフの作成
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=scaler.inverse_transform(df['y'].values.reshape(-1, 1)).flatten(), mode='lines', name='Actual Data', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df['ds'], y=trainPredictPlot, mode='lines', name='Train Prediction', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df['ds'], y=valPredictPlot, mode='lines', name='Validation Prediction', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df['ds'], y=testPredictPlot, mode='lines', name='Test Prediction', line=dict(color='red')))
fig.add_trace(go.Scatter(x=future_dates, y=futurePredictPlot, mode='lines', name='Future Prediction', line=dict(color='purple')))
fig.update_layout(title='USD/JPY Price Prediction using LSTM', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
fig.show()
