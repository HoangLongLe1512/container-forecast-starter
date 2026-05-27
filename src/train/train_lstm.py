from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

import joblib

# =========================
# ROOT PATH
# =========================
BASE_DIR = Path.cwd()

DATA_PATH = BASE_DIR / "resources" / "data" / "Dataset update.xlsx"
MODEL_PATH = BASE_DIR / "resources" / "models"
MODEL_PATH.mkdir(exist_ok=True, parents=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(DATA_PATH)

df = df[df["Loại hình"].astype(str).str.contains("cont", case=False, na=False)]

df["ATA_dt"] = pd.to_datetime(df["ATA ngày tàu đến thực tế"], errors="coerce")
df["Số lượng"] = pd.to_numeric(df["Số lượng"], errors="coerce")

# =========================
# MAP NGÀNH HÀNG
# =========================
def map_nganh(x):
    x = str(x).lower()
    if any(k in x for k in ["máy", "thiết bị", "bông", "hạt nhựa"]):
        return "Sản xuất công nghiệp"
    return "Khác"

df["Nhóm hàng"] = df["Tên hàng"].apply(map_nganh)

df = df[df["Nhóm hàng"] == "Sản xuất công nghiệp"]

# =========================
# TIME SERIES MONTHLY
# =========================
monthly = df.groupby(pd.Grouper(key="ATA_dt", freq="ME"))["Số lượng"].sum()
monthly = monthly.asfreq("ME").fillna(0)

series = np.log1p(monthly)

# =========================
# TRAIN / TEST SPLIT
# =========================
def split_time_series(
    series,
    test_ratio=0.2
):

    test_size = int(
        len(series) * test_ratio
    )

    train = series[:-test_size]
    test = series[-test_size:]

    return train, test

train, test = split_time_series(
    series
)

# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(
    data,
    seq_length
):

    X, y = [], []

    for i in range(
        len(data) - seq_length
    ):

        X.append(
            data[i:i + seq_length]
        )

        y.append(
            data[i + seq_length]
        )

    return np.array(X), np.array(y)

# =========================
# SCALER
# =========================
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(
    train.values.reshape(-1, 1)
)

# =========================
# LSTM DATA
# =========================
SEQ_LENGTH = 12

X_train, y_train = create_sequences(
    train_scaled,
    SEQ_LENGTH
)

# =========================
# BUILD MODEL
# =========================
model = Sequential([

    LSTM(
        32,
        input_shape=(SEQ_LENGTH, 1)
    ),

    Dropout(0.2),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# =========================
# EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor="loss",
    patience=10,
    restore_best_weights=True
)

# =========================
# TRAIN MODEL
# =========================
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=4,
    validation_split=0.2,
    shuffle=False
    verbose=1,
    callbacks=[early_stop]
)

# =========================
# SAVE MODEL
# =========================
model.save(
    MODEL_PATH / "lstm_model.keras"
)

joblib.dump(
    scaler,
    MODEL_PATH / "scaler.pkl"
)

joblib.dump(
    series,
    MODEL_PATH / "series.pkl"
)

joblib.dump(
    SEQ_LENGTH,
    MODEL_PATH / "seq_length.pkl"
)

print("Đã chạy xong")
