import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import os
import pandas as pd
import numpy as np
import pandas_ta as ta 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense

# Veri setini yükle
try:
    csv_file_path = 'btcusd_1-min_data.csv'
    df_original = pd.read_csv(csv_file_path)
    print(f"Veri seti '{csv_file_path}' başarıyla yüklendi.")
except FileNotFoundError:
    print(f"Hata: '{csv_file_path}' dosyası bulunamadı.")
    print("Lütfen dosya yolunu kontrol edin ve güncelleyin.")
    exit()

df_original['Timestamp'] = pd.to_datetime(df_original['Timestamp'], unit='s')
df_original.set_index('Timestamp', inplace=True)

print("Dakikalık veriler saatlik OHLCV formatına dönüştürülüyor...")
# Dakikalık veriyi saatlik olarak topla (aggregate). Bu, göstergeler için gereklidir.
ohlcv_agg = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}
df_hourly = df_original.resample('H').agg(ohlcv_agg)
df_hourly.dropna(inplace=True)

print("Teknik analiz göstergeleri hesaplanıyor...")
# pandas_ta kullanarak göstergeleri hesapla ve veri setine ekle
df_hourly.ta.sma(length=12, append=True, col_names=('SMA_12',))
df_hourly.ta.ema(length=24, append=True, col_names=('EMA_24',))
df_hourly.ta.rsi(length=14, append=True, col_names=('RSI_14',))
df_hourly.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))
bollinger = df_hourly.ta.bbands(length=20, append=True, col_names=('BBL_20', 'BBM_20', 'BBU_20', 'BBB_20', 'BBP_20'))

df_hourly['BB_width'] = df_hourly['BBU_20'] - df_hourly['BBL_20']

# Data for model
data_full = df_hourly.reset_index()

# Son 15000 saatlik veriyi kullan
total_recent_hours_to_use = 15000
fixed_test_size_hours = 3000
data = data_full.tail(total_recent_hours_to_use).reset_index(drop=True)

# Gecikme (lag) özellikleri oluştur
n_lags = 7
for lag in range(1, n_lags + 1):
    data[f'lag_{lag}'] = data['Close'].shift(lag)

# Oluşturulan özelliklerden kaynaklanan NaN değerlerini temizle
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)


lag_cols = [f'lag_{lag}' for lag in range(1, n_lags + 1)]
ta_cols = ['SMA_12', 'EMA_24', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BB_width']
X_cols = lag_cols + ta_cols

print("\nModelde kullanılacak özellikler:")
print(X_cols)


# Veriyi eğitim ve test setlerine ayır
X_orig = data[X_cols]
y_orig = data['Close']
test_size_points = fixed_test_size_hours
train_size_data = len(data) - test_size_points
X_train_orig, X_test_orig = X_orig.iloc[:train_size_data], X_orig.iloc[train_size_data:]
y_train_orig, y_test_orig = y_orig.iloc[:train_size_data], y_orig.iloc[train_size_data:]
test_dates = data['Timestamp'].iloc[train_size_data:].reset_index(drop=True)

# Veriyi ölçeklendir
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train = scaler_X.fit_transform(X_train_orig)
X_test = scaler_X.transform(X_test_orig)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train_orig.values.reshape(-1, 1))

print(f"\nEğitim veri boyutu: {len(X_train)}")
print(f"Test veri boyutu: {len(X_test)}")

X_train_lstm = X_train[:, :n_lags][:, ::-1].reshape(-1, n_lags, 1)
X_test_lstm = X_test[:, :n_lags][:, ::-1].reshape(-1, n_lags, 1)


LOAD_MODELS_FROM_FILE = False
MODELS_DIR = "saved_models"
if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

lr_model_path = os.path.join(MODELS_DIR, 'linear_regression_model.joblib')
knn_model_path = os.path.join(MODELS_DIR, 'knn_regression_model.joblib')
rf_model_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
lstm_model_path = os.path.join(MODELS_DIR, 'lstm_model.h5')

if LOAD_MODELS_FROM_FILE and all(os.path.exists(p) for p in [lr_model_path, knn_model_path, rf_model_path, lstm_model_path]):
    print("Kaydedilmiş modeller yükleniyor...")
    model_lr = joblib.load(lr_model_path)
    model_knn = joblib.load(knn_model_path)
    model_rf = joblib.load(rf_model_path)
    model_lstm = load_model(lstm_model_path)
else:
    print("Modeller eğitiliyor...")
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train_scaled.ravel())
    joblib.dump(model_lr, lr_model_path)

    model_knn = KNeighborsRegressor(n_neighbors=5)
    model_knn.fit(X_train, y_train_scaled.ravel())
    joblib.dump(model_knn, knn_model_path)

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train_scaled.ravel())
    joblib.dump(model_rf, rf_model_path)

    model_lstm = Sequential([LSTM(50, input_shape=(n_lags, 1)), Dense(1)])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_train_lstm, y_train_scaled, epochs=100, batch_size=32, verbose=1)
    model_lstm.save(lstm_model_path)

# Tahminleri yap
y_pred_scaled_lr = model_lr.predict(X_test)
y_pred_lr = scaler_y.inverse_transform(y_pred_scaled_lr.reshape(-1,1)).flatten()

y_pred_scaled_knn = model_knn.predict(X_test)
y_pred_knn = scaler_y.inverse_transform(y_pred_scaled_knn.reshape(-1,1)).flatten()

y_pred_scaled_rf = model_rf.predict(X_test)
y_pred_rf = scaler_y.inverse_transform(y_pred_scaled_rf.reshape(-1,1)).flatten()

y_pred_scaled_lstm = model_lstm.predict(X_test_lstm)
y_pred_lstm = scaler_y.inverse_transform(y_pred_scaled_lstm).flatten()

# Performans metriklerini hesapla (Bu kısım değişmedi)
def calculate_mape(y_true, y_pred):
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true_arr != 0
    return np.mean(np.abs((y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask])) * 100

models_metrics = {
    'Lineer Regresyon': {'pred': y_pred_lr},
    'KNN': {'pred': y_pred_knn},
    'Random Forest': {'pred': y_pred_rf},
    'LSTM': {'pred': y_pred_lstm}
}
for name, values in models_metrics.items():
    metrics = {'mae': mean_absolute_error(y_test_orig, values['pred']),
               'r2': r2_score(y_test_orig, values['pred']),
               'acc': 100 - calculate_mape(y_test_orig, values['pred'])}
    values.update(metrics)
    print(f"--- {name} Performansı (Teknik Analizli) ---")
    print(f"MAE: {metrics['mae']:,.0f}, R²: {metrics['r2']:.2f}, Doğruluk: {metrics['acc']:.2f}%")



plot_test_points_on_graph = min(len(test_dates), 200)
model_plot_info = {
    'Lineer Regresyon': {'pred': y_pred_lr, 'color': 'forestgreen', 'style': '--'},
    'KNN': {'pred': y_pred_knn, 'color': 'purple', 'style': ':'},
    'Random Forest': {'pred': y_pred_rf, 'color': 'crimson', 'style': '-.'},
    'LSTM': {'pred': y_pred_lstm, 'color': 'blue', 'style': '-'}
}

for model_name, plot_info in model_plot_info.items():
    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    metrics = models_metrics[model_name]
    ax.plot(test_dates.tail(plot_test_points_on_graph), y_test_orig.tail(plot_test_points_on_graph).values, label='Gerçek Fiyat', color='darkorange', linewidth=2, marker='o', markersize=4)
    ax.plot(test_dates.tail(plot_test_points_on_graph), plot_info['pred'][-plot_test_points_on_graph:], label=f'{model_name} Tahmini', color=plot_info['color'], linestyle=plot_info['style'], linewidth=2)
    ax.set_title(f'{model_name} Tahminleri ve Gerçek Fiyatlar', fontsize=16)
    ax.set_xlabel('Tarih', fontsize=12)
    ax.set_ylabel('Fiyat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10)
    metrics_text = (f"MAE: {metrics['mae']:,.0f}\n"
                      f"R²: {metrics['r2']:.2f}\n"
                      f"Doğruluk: {metrics['acc']:.2f}%")
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()




def future_forecast_sklearn_with_ta(model, X_test_last_row_scaled, scaler_x, scaler_y, n_lags_val, hours_to_forecast):
    last_features_scaled = X_test_last_row_scaled.copy().reshape(1, -1)
    future_preds_scaled = []
    lag_scalers = [{'min': scaler_x.min_[i], 'scale': scaler_x.scale_[i]} for i in range(n_lags_val)]
    for _ in range(hours_to_forecast):
        pred_y_scaled = model.predict(last_features_scaled)[0]
        future_preds_scaled.append(pred_y_scaled)
        pred_y_orig = scaler_y.inverse_transform([[pred_y_scaled]])[0, 0]
        new_lag1_scaled = np.clip((pred_y_orig - lag_scalers[0]['min']) * lag_scalers[0]['scale'], 0, 1)
        # Lag değerlerini kaydır ve en sona yeni tahmini ekle
        current_lags = last_features_scaled[0, :n_lags_val]
        new_lags = np.roll(current_lags, -1)
        new_lags[-1] = new_lag1_scaled
        last_features_scaled[0, :n_lags_val] = new_lags
        
    return future_preds_scaled


def future_forecast_lstm_custom(model, last_sequence, scaler_X, scaler_y, n_lags_val, hours_to_forecast):
    future_preds_scaled = []
    current_sequence = last_sequence.copy()
    min_X, scale_X = scaler_X.min_[0], scaler_X.scale_[0] # Sadece lag'lar için ölçekleyici
    for _ in range(hours_to_forecast):
        pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        future_preds_scaled.append(pred_scaled)
        pred_orig = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        new_val_scaled = np.clip((pred_orig - min_X) * scale_X, 0, 1)
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = new_val_scaled
    return future_preds_scaled

hours_to_forecast = 5
future_preds = {}
if len(X_test) > 0:
    X_test_last_row_scaled = X_test[-1, :]
    X_test_last_sequence_lstm = X_test_lstm[-1:, :, :]

    # Tahminleri al
    future_preds['Lineer Regresyon'] = scaler_y.inverse_transform(np.array(future_forecast_sklearn_with_ta(model_lr, X_test_last_row_scaled, scaler_X, scaler_y, n_lags, hours_to_forecast)).reshape(-1,1)).flatten()
    future_preds['KNN'] = scaler_y.inverse_transform(np.array(future_forecast_sklearn_with_ta(model_knn, X_test_last_row_scaled, scaler_X, scaler_y, n_lags, hours_to_forecast)).reshape(-1,1)).flatten()
    future_preds['Random Forest'] = scaler_y.inverse_transform(np.array(future_forecast_sklearn_with_ta(model_rf, X_test_last_row_scaled, scaler_X, scaler_y, n_lags, hours_to_forecast)).reshape(-1,1)).flatten()
    future_preds['LSTM'] = scaler_y.inverse_transform(np.array(future_forecast_lstm_custom(model_lstm, X_test_last_sequence_lstm, scaler_X, scaler_y, n_lags, hours_to_forecast)).reshape(-1,1)).flatten()


forecast_index = np.arange(1, hours_to_forecast + 1)
plt.figure(figsize=(14, 8))
ax2 = plt.gca()
for model_name, plot_info in model_plot_info.items():
    ax2.plot(forecast_index, future_preds[model_name], label=f'{model_name} Tahmini', color=plot_info['color'], linestyle=plot_info['style'], marker='o' if plot_info['style'] == '-' else '^')
ax2.set_title(f'BTC Fiyat Tahmini', fontsize=16)
ax2.set_ylabel('Tahmini Fiyat', fontsize=12)
ax2.grid(True, alpha=0.6)
ax2.legend(loc='best')
ax2.set_xticks(forecast_index)
ax2.xaxis.set_ticklabels([])
ax2.tick_params(axis='x', which='both', length=0)
ax2.set_xlabel('')
for model_name, plot_info in model_plot_info.items():
    for i, price in enumerate(future_preds[model_name]):
        ax2.text(forecast_index[i], price, f'{price:,.0f}', ha='center', va='bottom', fontsize=9, color=plot_info['color'], bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
plt.tight_layout()
plt.show()
