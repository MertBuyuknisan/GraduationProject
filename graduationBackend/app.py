import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_data_and_models():
    print("Sunucu başlatılıyor: Veri işleniyor ve modeller yükleniyor...")
    try:
        csv_file_path = 'btcusd_1-min_data.csv'
        df_original = pd.read_csv(csv_file_path, low_memory=False)
    except FileNotFoundError:
        print(f"HATA: '{csv_file_path}' dosyası bulunamadı!")
        exit()

    df_original['Timestamp'] = pd.to_datetime(df_original['Timestamp'], unit='s')
    df_original.set_index('Timestamp', inplace=True)
    ohlcv_agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    df_hourly = df_original.resample('H').agg(ohlcv_agg)
    df_hourly.dropna(inplace=True)
    df_hourly.ta.sma(length=12, append=True, col_names=('SMA_12',))
    df_hourly.ta.ema(length=24, append=True, col_names=('EMA_24',))
    df_hourly.ta.rsi(length=14, append=True, col_names=('RSI_14',))
    df_hourly.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))
    df_hourly.ta.bbands(length=20, append=True, col_names=('BBL_20', 'BBM_20', 'BBU_20', 'BBB_20', 'BBP_20'))
    df_hourly['BB_width'] = df_hourly['BBU_20'] - df_hourly['BBL_20']
    
    data_full = df_hourly.reset_index()
    total_recent_hours_to_use = 15000
    fixed_test_size_hours = 3000
    data = data_full.tail(total_recent_hours_to_use).reset_index(drop=True)
    n_lags = 7
    for lag in range(1, n_lags + 1):
        data[f'lag_{lag}'] = data['Close'].shift(lag)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    lag_cols = [f'lag_{lag}' for lag in range(1, n_lags + 1)]
    ta_cols = ['SMA_12', 'EMA_24', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BB_width']
    X_cols = lag_cols + ta_cols

    X_orig = data[X_cols]
    y_orig = data['Close']
    test_size_points = fixed_test_size_hours
    train_size_data = len(data) - test_size_points
    X_train_orig, X_test_orig = X_orig.iloc[:train_size_data], X_orig.iloc[train_size_data:]
    y_train_orig, y_test_orig = y_orig.iloc[:train_size_data], y_orig.iloc[train_size_data:]
    test_dates = data['Timestamp'].iloc[train_size_data:].reset_index(drop=True)

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train_orig)
    X_test = scaler_X.transform(X_test_orig)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler_y.fit_transform(y_train_orig.values.reshape(-1, 1))
    X_test_lstm = X_test[:, :n_lags][:, ::-1].reshape(-1, n_lags, 1)

    MODELS_DIR = "saved_models"
    lr_model_path = os.path.join(MODELS_DIR, 'linear_regression_model.joblib')
    knn_model_path = os.path.join(MODELS_DIR, 'knn_regression_model.joblib')
    rf_model_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
    lstm_model_path = os.path.join(MODELS_DIR, 'lstm_model.h5')
    
    if not all(os.path.exists(p) for p in [lr_model_path, knn_model_path, rf_model_path, lstm_model_path]):
        print("HATA: Kaydedilmiş modeller 'saved_models' klasöründe bulunamadı.")
        exit()

    model_lr = joblib.load(lr_model_path)
    model_knn = joblib.load(knn_model_path)
    model_rf = joblib.load(rf_model_path)
    model_lstm = load_model(lstm_model_path)

    y_pred_lr = scaler_y.inverse_transform(model_lr.predict(X_test).reshape(-1,1)).flatten()
    y_pred_knn = scaler_y.inverse_transform(model_knn.predict(X_test).reshape(-1,1)).flatten()
    y_pred_rf = scaler_y.inverse_transform(model_rf.predict(X_test).reshape(-1,1)).flatten()
    y_pred_lstm = scaler_y.inverse_transform(model_lstm.predict(X_test_lstm)).flatten()

    plot_points = 200
    
    global PREDICTION_DATA
    PREDICTION_DATA = {
        "test_dates": [d.isoformat() for d in test_dates.tail(plot_points).tolist()],
        "actual_prices": y_test_orig.tail(plot_points).tolist(),
        "predictions": {
            "Lineer Regresyon": y_pred_lr[-plot_points:].tolist(),
            "KNN": y_pred_knn[-plot_points:].tolist(),
            "Random Forest": y_pred_rf[-plot_points:].tolist(),
            "LSTM": y_pred_lstm[-plot_points:].tolist()
        }
    }
    print("Veri ve modeller başarıyla yüklendi. Sunucu istekleri kabul etmeye hazır.")

@app.route('/predict', methods=['GET'])
def get_predictions():
    if PREDICTION_DATA is None:
        return jsonify({"error": "Sunucu verisi hazır değil."}), 500
    return jsonify(PREDICTION_DATA)

if __name__ == '__main__':
    PREDICTION_DATA = None
    load_data_and_models()
    app.run(host='0.0.0.0', port=5000)