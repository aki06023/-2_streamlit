import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# モデルとスケーラーの読み込み
model = load_model('lstm_usdjpy_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

def preprocess_and_predict(symbol: str):
    df = yf.download(symbol, start='2015-01-01', end='2024-07-31')
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['y'] = scaler.fit_transform(np.array(df['y']).reshape(-1, 1))

    look_back = 30

    def create_dataset(data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back):
            a = data[i:(i + look_back), 0]
            X.append(a)
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    data = df['y'].values.reshape(-1, 1)
    X, _ = create_dataset(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    future_predictions = []
    last_input = X[-1]

    for _ in range(30):
        prediction = model.predict(last_input.reshape(1, look_back, 1))
        future_predictions.append(prediction[0][0])
        last_input = np.append(last_input[1:], prediction, axis=0)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    last_date = df['ds'].values[-1]
    future_dates = pd.date_range(start=last_date, periods=len(future_predictions), freq='D')

    actual_data = scaler.inverse_transform(df['y'].values.reshape(-1, 1)).flatten()
    trainPredictPlot = np.empty_like(actual_data)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:] = actual_data[:len(trainPredictPlot) - look_back]

    futurePredictPlot = np.empty(len(future_dates))
    futurePredictPlot[:] = np.nan
    futurePredictPlot[:len(future_predictions)] = future_predictions.flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=actual_data, mode='lines', name='Actual Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=futurePredictPlot, mode='lines', name='Future Prediction', line=dict(color='purple')))
    fig.update_layout(title='Price Prediction using LSTM', xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    return {
        'future_predictions': future_predictions.flatten().tolist(),
        'future_dates': future_dates.strftime('%Y-%m-%d').tolist(),
        'plot': fig.to_html(full_html=False)
    }

# Streamlitアプリケーション
st.title('価格予測アプリ')

# ドロップダウンリストの作成
symbols = ['USDJPY=X','CADJPY=X','AUDJPY=X','EURJPY=X','GBPJPY=X','EURUSD=X','GBPUSD=X']  # ここにリストを追加
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

    st.write('予測グラフ:')
    st.components.v1.html(results['plot'], height=600)

