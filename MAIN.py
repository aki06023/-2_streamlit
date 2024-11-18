import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import jpholiday  # 日本の祝日を除外するために使用

# モデルとスケーラーの読み込み
model = load_model('lstm_usdjpy_model (1).h5')
scaler = MinMaxScaler(feature_range=(0, 1))

def is_holiday(date):
    """日本の土日祝日を判定"""
    return date.weekday() >= 5 or jpholiday.is_holiday(date)

def preprocess_and_predict(symbol: str):
    df = yf.download(symbol, start='2015-01-01', end='2024-11-08')
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['y'] = scaler.fit_transform(np.array(df['y']).reshape(-1, 1))

    look_back = 30

    def create_dataset(data, look_back=1):
        X, Y = [], []  # リストをタプル形式で初期化
        for i in range(len(data) - look_back):
            a = data[i:(i + look_back), 0]
            X.append(a)
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    data = df['y'].values.reshape(-1, 1)
    X, y = create_dataset(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # スコア計算用データ分割
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)
    trainX, valX, testX = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    trainY, valY, testY = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    # 予測
    trainPredict = model.predict(trainX)
    valPredict = model.predict(valX)
    testPredict = model.predict(testX)

    # 逆正規化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
    valPredict = scaler.inverse_transform(valPredict)
    valY = scaler.inverse_transform(valY.reshape(-1, 1))
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY.reshape(-1, 1))

    # スコア計算
    trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
    valScore = np.sqrt(mean_squared_error(valY, valPredict))
    testScore = np.sqrt(mean_squared_error(testY, testPredict))

    # 未来予測
    future_predictions = []
    last_input = X[-1]

    for _ in range(30):
        prediction = model.predict(last_input.reshape(1, look_back, 1))
        future_predictions.append(prediction[0][0])
        last_input = np.append(last_input[1:], prediction, axis=0)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # 土日祝日を除外した日付を生成
    last_date = pd.to_datetime(df['ds'].iloc[-1])
    future_dates = []
    current_date = last_date
    while len(future_dates) < len(future_predictions):
        current_date += pd.Timedelta(days=1)
        if not is_holiday(current_date):
            future_dates.append(current_date)

    actual_data = scaler.inverse_transform(df['y'].values.reshape(-1, 1)).flatten()

    # プロットの準備
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=actual_data, mode='lines', name='Actual Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Future Prediction', line=dict(color='purple')))
    fig.update_layout(title='Price Prediction (Excluding Holidays)', xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    return {
        'future_predictions': future_predictions.flatten().tolist(),
        'future_dates': [date.strftime('%Y-%m-%d') for date in future_dates],
        'trainScore': trainScore,
        'valScore': valScore,
        'testScore': testScore,
        'plot': fig.to_html(full_html=False)
    }

# Streamlitアプリケーション
st.title('Predict Future Prices Excluding Holidays')

# ドロップダウンリストの作成
symbols = ['USDJPY=X', 'CADJPY=X', 'AUDJPY=X', 'EURJPY=X', 'GBPJPY=X', 'EURUSD=X', 'GBPUSD=X']
symbol = st.selectbox('シンボルを選択してください', symbols)

if symbol:
    st.write(f'選択したシンボル: {symbol}')
    results = preprocess_and_predict(symbol)

    st.write('未来の予測価格:')
    predictions_df = pd.DataFrame({
        'Date': results['future_dates'],
        'Prediction': results['future_predictions']
    })
    st.write(predictions_df)

    st.write("### RMSE Scores")
    st.write(f"Train Score: {results['trainScore']:.2f} RMSE")
    st.write(f"Validation Score: {results['valScore']:.2f} RMSE")
    st.write(f"Test Score: {results['testScore']:.2f} RMSE")

    st.write('予測グラフ:')
    st.components.v1.html(results['plot'], height=600)
