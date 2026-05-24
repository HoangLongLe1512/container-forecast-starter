import os
import joblib
import pandas as pd
import numpy as np

from django.http import JsonResponse
from django.shortcuts import render

from tensorflow.keras.models import load_model

# =========================
# BASE PATH
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "resources",
    "models"
)

# =========================
# LOAD MODEL
# =========================
model = load_model(
    os.path.join(
        MODEL_PATH,
        "lstm_model.keras"
    )
)

scaler = joblib.load(
    os.path.join(
        MODEL_PATH,
        "scaler.pkl"
    )
)

series = joblib.load(
    os.path.join(
        MODEL_PATH,
        "series.pkl"
    )
)

seq_length = joblib.load(
    os.path.join(
        MODEL_PATH,
        "seq_length.pkl"
    )
)

# =========================
# FORECAST VIEW
# =========================
def forecast(request):

    n = int(
        request.GET.get("n", 6)
    )

    history = series.copy()

    preds = []

    # =========================
    # LAST SEQUENCE
    # =========================
    last_values = history.tail(
        seq_length
    )

    current_seq = scaler.transform(
        last_values.values.reshape(-1, 1)
    )

    # =========================
    # RECURSIVE FORECAST
    # =========================
    for _ in range(n):

        pred_scaled = model.predict(
            current_seq.reshape(
                1,
                seq_length,
                1
            ),
            verbose=0
        )[0][0]

        # inverse scale
        pred_log = scaler.inverse_transform(
            np.array(pred_scaled)
            .reshape(-1, 1)
        )[0][0]

        # next month
        next_date = (
            history.index[-1]
            + pd.offsets.MonthEnd(1)
        )

        # append history
        history.loc[next_date] = pred_log

        # save result
        preds.append({

            "month": next_date.strftime(
                "%Y-%m"
            ),

            "forecast": round(
                float(np.expm1(pred_log)),
                0
            )
        })

        # update sequence
        current_seq = np.append(
            current_seq[1:],
            [[pred_scaled]],
            axis=0
        )

    return JsonResponse({
        "predictions": preds
    })

# =========================
# HOME PAGE
# =========================
def home(request):

    return render(
        request,
        "index.html"
    )