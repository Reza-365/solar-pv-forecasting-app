import os
# Community Cloud currently may offer Python 3.14, but the saved TensorFlow
# runtime in this project is TensorFlow 2.20, which has no CPython 3.14 wheel.
# Keep this check early so a wrong deployment shows a useful message.
if os.sys.version_info[:2] not in {(3, 12), (3, 13)}:
    raise RuntimeError(
        f"Unsupported Python runtime: {os.sys.version.split()[0]}. "
        "This deployment is intentionally pinned to Python 3.12/3.13 because "
        "the saved Keras models require a TensorFlow wheel available for those runtimes. "
        "In Streamlit Community Cloud, delete this app and redeploy it with Python 3.12 "
        "under Advanced settings."
    )

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Limit TensorFlow thread creation on Streamlit Cloud to reduce resource pressure.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

from pathlib import Path
from urllib.parse import quote
import json
import time

# ---------------------------------------------------------------------------
# Keras custom-object compatibility
# The saved .keras models were trained with a custom `physics_loss` function.
# Keras stores the function name in the model config, not its Python source.
# It therefore must exist before load_model() is called.  For this live app we
# only need the trained weights for inference, so compile=False is intentional.
# ---------------------------------------------------------------------------
def physics_loss(y_true, y_pred):
    """Inference-time fallback for the custom loss used during training.

    The saved models are loaded with compile=False, so this function is not
    used to calculate forecasts. It is supplied to Keras only so older model
    metadata referring to `physics_loss` can be deserialized safely.
    """
    import tensorflow as tf
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    negative_penalty = tf.reduce_mean(tf.nn.relu(-y_pred))
    return mse + 0.05 * negative_penalty

try:
    import keras as _keras_for_registration
    physics_loss = _keras_for_registration.saving.register_keras_serializable(
        package="SolarPV", name="physics_loss"
    )(physics_loss)
except Exception:
    pass

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
BUNDLE = BASE / "model_bundle"
MODEL_DIR = BUNDLE / "models"
CFG_DIR = BUNDLE / "configuration"
Q_DIR = BUNDLE / "q_learning"
METRIC_DIR = BUNDLE / "metrics"

CONFIG = json.loads((CFG_DIR / "configuration.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((CFG_DIR / "manifest.json").read_text(encoding="utf-8"))

TARGET = MANIFEST["target"]
FEATURE_COLUMNS = MANIFEST["feature_columns"]
NPAST = int(MANIFEST["npast"])
HORIZON = int(MANIFEST["future_horizon"])
LAT_DEG = float(CONFIG.get("site_latitude_deg", 23.9999))
INTERVAL_MIN = int(CONFIG.get("sampling_interval_minutes", 5))
CAL_FACTOR = float(MANIFEST.get("PI_calibration_factor", 0.5))
SELECTED = MANIFEST["selected_models"]

st.set_page_config(
    page_title="Solar PV Live Forecast",
    page_icon="☀️",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading trained models...")
def load_artifacts():
    import joblib
    import keras

    custom_objects = {"physics_loss": physics_loss}

    def load_keras_for_inference(path):
        """Load a saved Keras model without recompiling its training graph."""
        try:
            return keras.models.load_model(
                path,
                custom_objects=custom_objects,
                compile=False,
                safe_mode=False,
            )
        except TypeError:
            # Compatibility with older Keras signatures.
            return keras.models.load_model(
                path,
                custom_objects=custom_objects,
                compile=False,
            )

    scalers = joblib.load(CFG_DIR / "scalers.joblib")
    imputer = scalers["imputer"]
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]

    trained = {}
    for name in SELECTED:
        if name in {"LR", "KNN", "SVR", "XGBoost", "LightGBM"}:
            trained[name] = joblib.load(MODEL_DIR / f"{name}.joblib")
        elif name == "LSTM":
            trained[name] = load_keras_for_inference(MODEL_DIR / "LSTM.keras")
        elif name == "LSTM_XGB":
            # The saved feature model is a full LSTM -> Dense(1) model, while
            # the companion XGBoost model was trained on the 64-dimensional
            # LSTM hidden representation.  Do NOT feed the Dense(1) prediction
            # into XGBoost (that causes: expected 64, got 1).
            full_model = load_keras_for_inference(
                MODEL_DIR / "LSTM_XGB_feature_model.keras"
            )
            try:
                feature_layer = full_model.get_layer("feature_lstm")
            except Exception:
                # Fallback: find the LSTM layer with 64 units.
                feature_layer = next(
                    layer for layer in full_model.layers
                    if hasattr(layer, "units") and int(layer.units) == 64
                )
            feature_extractor = keras.Model(
                inputs=full_model.inputs,
                outputs=feature_layer.output,
                name="LSTM_XGB_64D_feature_extractor",
            )
            trained[name] = {
                "extractor": feature_extractor,
                "xgb": joblib.load(MODEL_DIR / "LSTM_XGB_xgb.joblib"),
            }

    stacker = joblib.load(MODEL_DIR / "dynamic_stacker.joblib")
    q_scores = pd.read_csv(Q_DIR / "q_learning_model_scores.csv").set_index("Model")["QScore"].to_dict()
    assessment = pd.read_csv(METRIC_DIR / "candidate_assessment.csv").set_index("Model")
    return trained, stacker, q_scores, assessment, imputer, x_scaler, y_scaler

def normalize_columns(df):
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip().str.replace(" ", "_", regex=False)
    return out

def resolve_target(columns, requested):
    req = str(requested).strip().replace(" ", "_")
    if req in columns:
        return req
    compact = req.lower().replace("Ω", "ohm")
    for c in columns:
        if c.lower().replace("Ω", "ohm") == compact:
            return c
    candidates = [c for c in columns if "load_power" in c.lower()]
    if candidates:
        return candidates[0]
    raise KeyError(f"Target '{requested}' was not found.")

def prepare_dataframe(df_raw):
    df = normalize_columns(df_raw)
    target = resolve_target(df.columns, CONFIG["target"])

    if "Date" in df.columns and "Time" in df.columns:
        dt = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
        )
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        raise KeyError("The Google Sheet must contain Date and Time columns.")

    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df = df.drop_duplicates("datetime").reset_index(drop=True)

    tf = pd.DataFrame(index=df.index)
    tf["hour"] = df["datetime"].dt.hour
    tf["minute"] = df["datetime"].dt.minute
    tf["day"] = df["datetime"].dt.day
    tf["month"] = df["datetime"].dt.month
    tf["dayofweek"] = df["datetime"].dt.dayofweek
    tf["dayofyear"] = df["datetime"].dt.dayofyear
    tf["hour_sin"] = np.sin(2 * np.pi * (tf["hour"] + tf["minute"] / 60.0) / 24)
    tf["hour_cos"] = np.cos(2 * np.pi * (tf["hour"] + tf["minute"] / 60.0) / 24)
    tf["doy_sin"] = np.sin(2 * np.pi * tf["dayofyear"] / 365.25)
    tf["doy_cos"] = np.cos(2 * np.pi * tf["dayofyear"] / 365.25)

    work = df.drop(columns=["Date", "Time", "datetime"], errors="ignore").copy()
    for c in work.columns:
        if work[c].dtype == object:
            work[c] = (
                work[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
                .replace({
                    "": np.nan, "nan": np.nan, "None": np.nan,
                    "true": 1, "True": 1, "TRUE": 1,
                    "false": 0, "False": 0, "FALSE": 0,
                })
            )
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(axis=1, how="all")

    if target not in work.columns:
        target = resolve_target(work.columns, CONFIG["target"])

    work = work.interpolate(limit_direction="both").ffill().bfill()

    for c in tf.columns:
        work[c] = tf[c].values

    for lag in [1, 3, 6, 12, 24]:
        work[f"{target}_lag_{lag}"] = work[target].shift(lag)

    valid = work.notna().all(axis=1)
    dt_clean = df.loc[work.index[valid], "datetime"].reset_index(drop=True)
    work = work.loc[valid].reset_index(drop=True)

    return work, dt_clean, target

def daylight_factor(dt, lat_deg):
    n = dt.timetuple().tm_yday
    decl = np.radians(23.44 * np.sin(np.radians(360.0 / 365.0 * (284 + n))))
    lat = np.radians(lat_deg)
    solar_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    hour_angle = np.radians(15.0 * (solar_hour - 12.0))
    cos_zenith = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    )
    return float(max(0.0, cos_zenith))

@st.cache_data(ttl=60, show_spinner=False)
def load_sheet():
    # Preferred for Streamlit deployment: public/readable Google Sheet via gviz.
    sheet_id = st.secrets.get("SHEET_ID", CONFIG["sheet_id"])
    worksheet = st.secrets.get("WORKSHEET_NAME", CONFIG["worksheet_name"])
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={quote(str(worksheet), safe='')}"
    )

    try:
        return pd.read_csv(url), f"Google Sheets public CSV ({worksheet})"
    except Exception as public_error:
        # Optional private-sheet route using a Streamlit service-account secret.
        try:
            import gspread
            from google.oauth2.service_account import Credentials

            if "gcp_service_account" not in st.secrets:
                raise public_error

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets.readonly",
                "https://www.googleapis.com/auth/drive.readonly",
            ]
            creds = Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]),
                scopes=scopes,
            )
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(sheet_id)
            ws = sh.worksheet(worksheet)
            values = ws.get_all_values()
            if not values:
                raise RuntimeError("The Google Sheet is empty.")
            return pd.DataFrame(values[1:], columns=values[0]), f"Google Sheets private API ({worksheet})"
        except Exception as private_error:
            raise RuntimeError(
                "Could not read the Google Sheet. Either make the sheet readable "
                "to the public link and keep the configured Sheet ID, or add a "
                "gcp_service_account object to Streamlit Secrets."
            ) from private_error

def make_forecast(raw_df, trained, stacker, q_scores, assessment, imputer, x_scaler, y_scaler):
    history, dts, target = prepare_dataframe(raw_df)

    missing = [c for c in FEATURE_COLUMNS if c not in history.columns]
    if missing:
        raise ValueError(
            "The Google Sheet is missing required model features: "
            + ", ".join(missing)
        )
    if len(history) < NPAST:
        raise ValueError(
            f"At least {NPAST} usable rows are required after feature engineering; "
            f"only {len(history)} are available."
        )

    history = history.copy()
    last_dt = dts.iloc[-1]
    last_target = float(history[target].iloc[-1])

    # Exact feature order used during training.
    X = history[FEATURE_COLUMNS].astype(float).values
    Xs = x_scaler.transform(imputer.transform(X))

    # The saved uncertainty calibration is derived from validation.
    median_unc = {}
    for m in SELECTED:
        if m in assessment.index and "Median_Uncertainty" in assessment.columns:
            median_unc[m] = float(assessment.loc[m, "Median_Uncertainty"])
        else:
            median_unc[m] = 1.0

    irr_col = next(
        (c for c in history.columns if "POA" in c or "irred" in c.lower() or "irradi" in c.lower()),
        None,
    )

    persist_cols = [
        c for c in history.columns
        if c not in [
            target, "hour", "minute", "day", "month", "dayofweek",
            "dayofyear", "hour_sin", "hour_cos", "doy_sin", "doy_cos"
        ]
    ]
    solar_keywords = ("irrad", "poa", "light", "lux", "solar")
    solar_cols = sorted(
        set(([irr_col] if irr_col else []) +
            [c for c in persist_cols if any(k in c.lower() for k in solar_keywords)])
        - {None}
    )

    origin_shape = max(daylight_factor(last_dt, LAT_DEG), 0.05)
    origin_solar = {
        c: float(history[c].iloc[-1])
        for c in solar_cols if c in history.columns
    }
    solar_clip = {
        c: float(history[c].max()) * 1.15
        for c in solar_cols if c in history.columns
    }

    rolling = history.copy()
    records = []

    def inv_y(z):
        return y_scaler.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()

    for step in range(1, HORIZON + 1):
        current_dt = last_dt + pd.Timedelta(minutes=INTERVAL_MIN * step)

        row = {
            c: float(rolling[c].iloc[-1])
            for c in persist_cols if c in rolling.columns
        }
        row[target] = last_target
        row["hour"] = current_dt.hour
        row["minute"] = current_dt.minute
        row["day"] = current_dt.day
        row["month"] = current_dt.month
        row["dayofweek"] = current_dt.dayofweek
        row["dayofyear"] = current_dt.dayofyear
        row["hour_sin"] = np.sin(
            2 * np.pi * (current_dt.hour + current_dt.minute / 60.0) / 24.0
        )
        row["hour_cos"] = np.cos(
            2 * np.pi * (current_dt.hour + current_dt.minute / 60.0) / 24.0
        )
        row["doy_sin"] = np.sin(2 * np.pi * current_dt.dayofyear / 365.25)
        row["doy_cos"] = np.cos(2 * np.pi * current_dt.dayofyear / 365.25)

        shape_t = daylight_factor(current_dt, LAT_DEG)
        solar_ratio = min(shape_t / origin_shape, 6.0)
        for c in solar_cols:
            if c in origin_solar:
                projected = origin_solar[c] * solar_ratio
                row[c] = float(np.clip(projected, 0.0, solar_clip.get(c, projected)))

        for lag in [1, 3, 6, 12, 24]:
            row[f"{target}_lag_{lag}"] = (
                float(rolling[target].iloc[-lag])
                if len(rolling) >= lag else last_target
            )

        row_df = pd.DataFrame([row])
        for c in FEATURE_COLUMNS:
            if c not in row_df.columns:
                row_df[c] = float(history[c].iloc[-1])
        row_df = row_df[FEATURE_COLUMNS]

        row_scaled = x_scaler.transform(imputer.transform(row_df))
        hist_window = rolling.tail(NPAST - 1)
        hist_part = x_scaler.transform(
            imputer.transform(hist_window[FEATURE_COLUMNS].values.astype(float))
        )
        seq = np.vstack([hist_part, row_scaled]).reshape(1, NPAST, len(FEATURE_COLUMNS))
        tab_input = row_scaled.reshape(1, -1)

        pvals, uvals = {}, {}

        for m in SELECTED:
            if m in {"LR", "KNN", "SVR", "XGBoost", "LightGBM"}:
                p = float(inv_y([trained[m].predict(tab_input)[0]])[0])
                u = median_unc[m]

            elif m == "LSTM_XGB":
                feat = trained[m]["extractor"].predict(seq, verbose=0)
                feat = np.asarray(feat, dtype=np.float32).reshape(1, -1)
                expected = int(getattr(trained[m]["xgb"], "n_features_in_", feat.shape[1]))
                if feat.shape[1] != expected:
                    raise RuntimeError(
                        f"LSTM-XGB feature mismatch: XGBoost expects {expected} features, "
                        f"but the LSTM extractor produced {feat.shape[1]}."
                    )
                p = float(inv_y([trained[m]["xgb"].predict(feat)[0]])[0])
                u = median_unc[m]

            elif m == "LSTM":
                model = trained[m]
                p = float(inv_y([model.predict(seq, verbose=0).ravel()[0]])[0])

                # Same MC-dropout idea as training, with the saved model.
                mc = np.asarray([
                    model(seq, training=True).numpy().ravel()[0]
                    for _ in range(int(CONFIG.get("mc_samples", 40)))
                ])
                # Convert MC-dropout spread from scaled target units back to mW.
                y_scale = float(np.asarray(getattr(y_scaler, "scale_", [1.0])).ravel()[0])
                mc_std = float(np.std(mc) * abs(y_scale))
                u = float(np.sqrt(max(mc_std, 0.0) ** 2 + median_unc[m] ** 2))

            else:
                raise ValueError(f"Unsupported selected model: {m}")

            pvals[m] = max(0.0, p)
            uvals[m] = max(1e-6, u)

        P = np.asarray([pvals[m] for m in SELECTED], dtype=float)
        U = np.asarray([uvals[m] for m in SELECTED], dtype=float)
        qv = np.asarray([q_scores.get(m, 0.5) for m in SELECTED], dtype=float)

        q_weights = qv / (U + 1e-6)
        q_weights = q_weights / (q_weights.sum() + 1e-12)
        prior_prediction = float(np.sum(P * q_weights))

        # 38 stacker features:
        # P(7), U(7), 7 aggregate/prior features, q-weights(7), context(10).
        context = []
        for c in ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "hour", "minute"]:
            context.append(float(row[c]))
        for c in [irr_col, f"{target}_lag_1", f"{target}_lag_3", f"{target}_lag_24"]:
            if c is not None and c in row:
                context.append(float(pd.to_numeric(pd.Series([row[c]]), errors="coerce").fillna(0).iloc[0]))

        stack_features = np.concatenate([
            P,
            U,
            np.array([
                P.mean(), P.std(), P.min(), P.max(),
                U.mean(), U.std(), prior_prediction
            ]),
            q_weights,
            np.asarray(context, dtype=float),
        ]).reshape(1, -1)

        if stack_features.shape[1] != getattr(stacker, "n_features_in_", 38):
            raise RuntimeError(
                f"Dynamic stacker expected {getattr(stacker, 'n_features_in_', 'unknown')} "
                f"features but the live feature builder produced {stack_features.shape[1]}."
            )

        stack_pred = float(stacker.predict(stack_features)[0])
        dyn_pred = max(0.0, 0.80 * stack_pred + 0.20 * prior_prediction)

        disagreement = float(np.std(P))
        within = float(np.sqrt(np.sum((q_weights * U) ** 2)))
        sigma = float(np.sqrt(disagreement ** 2 + within ** 2) * CAL_FACTOR)

        # Keep recursion continuous, but expose only the configured operating window.
        day_start = current_dt.normalize() + pd.Timedelta(
            hours=int(CONFIG["operating_window_start"].split(":")[0]),
            minutes=int(CONFIG["operating_window_start"].split(":")[1]),
        )
        day_end = current_dt.normalize() + pd.Timedelta(
            hours=int(CONFIG["operating_window_end"].split(":")[0]),
            minutes=int(CONFIG["operating_window_end"].split(":")[1]),
        )
        in_window = day_start <= current_dt <= day_end

        if (not CONFIG.get("operating_window_only", True)) or in_window:
            records.append({
                "step_ahead": step,
                "datetime": current_dt,
                "forecast_pv_power_mW": dyn_pred,
                "lower_95_mW": max(0.0, dyn_pred - 1.96 * sigma),
                "upper_95_mW": dyn_pred + 1.96 * sigma,
                "uncertainty_mW": sigma,
                "model_disagreement_mW": disagreement,
                "within_model_uncertainty_mW": within,
                "q_uncertainty_prior_prediction_mW": prior_prediction,
                "solar_daylight_factor": shape_t,
            })

        newrow = row.copy()
        newrow[target] = dyn_pred
        rolling = pd.concat([rolling, pd.DataFrame([newrow])], ignore_index=True)
        last_target = dyn_pred

    return pd.DataFrame(records), history, target

def main():
    st.title("☀️ Solar PV Live Forecast")
    st.caption(
        "Using Q-learning uncertainty-aware dynamic stacking models and IoT based data logger. "
        "© S. M. Rezaul Karim & Prof. Dr. Md. Monirul Kabir, "
        "Dhaka University of Engineering & Technology (DUET)"
    )

    try:
        trained, stacker, q_scores, assessment, imputer, x_scaler, y_scaler = load_artifacts()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = 0.0

    with st.sidebar:
        st.header("Live settings")
        auto_refresh = st.checkbox("Auto refresh", value=False)
        refresh_seconds = st.slider("Refresh interval (seconds)", 30, 300, 60, 15)
        if auto_refresh:
            st.caption(f"Dashboard refreshes every {refresh_seconds} seconds.")

    # Streamlit 1.64 supports timed fragments. The whole forecast block is
    # wrapped below so a live Google Sheet is actually re-read automatically.

    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast horizon", f"{HORIZON} steps / 2 hours")
    c2.metric("Selected models", str(len(SELECTED)))
    c3.metric("Sampling", f"{INTERVAL_MIN} min")

    if st.button("🔄 Refresh Google Sheet & Forecast", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # Auto-refresh is implemented as a timed fragment. This avoids relying on
    # browser interaction to trigger a rerun.
    @st.fragment(run_every=refresh_seconds if auto_refresh else None)
    def live_forecast_panel():
        if auto_refresh:
            st.cache_data.clear()

        try:
            raw_df, source = load_sheet()
            forecast, history, target = make_forecast(
                raw_df, trained, stacker, q_scores, assessment,
                imputer, x_scaler, y_scaler
            )
        except Exception as e:
            st.error(str(e))
            st.info(
                "For the simplest deployment, make the Google Sheet accessible by link. "
                "For a private sheet, add a Google service-account JSON object to Streamlit Secrets."
            )
            st.stop()

        st.success(f"Live data source: {source}")

        latest = history.iloc[-1]
        # history was already prepared by make_forecast(); obtain the final timestamp
        # from the raw sheet only once more without repeating feature engineering.
        normalized_raw = normalize_columns(raw_df)
        if "Date" in normalized_raw.columns and "Time" in normalized_raw.columns:
            latest_dt = pd.to_datetime(
                normalized_raw["Date"].astype(str) + " " + normalized_raw["Time"].astype(str),
                errors="coerce",
            ).dropna().max()
        elif "datetime" in normalized_raw.columns:
            latest_dt = pd.to_datetime(
                normalized_raw["datetime"], errors="coerce"
            ).dropna().max()
        else:
            latest_dt = pd.NaT

        a, b, c, d = st.columns(4)
        a.metric("Latest PV power", f"{float(latest[target]):,.0f} mW")
        b.metric("Latest observation", latest_dt.strftime("%Y-%m-%d %H:%M"))
        c.metric("Rows used", f"{len(history):,}")
        d.metric("Next forecast", f"{forecast.iloc[0]['forecast_pv_power_mW']:,.0f} mW" if len(forecast) else "Outside 06:00-18:00")

        st.subheader("Next 2-hour PV forecast")

        if forecast.empty:
            st.warning("The next 24 recursive steps fall outside the configured 06:00-18:00 operating window.")
        else:
            chart = forecast.set_index("datetime")[
                ["forecast_pv_power_mW", "lower_95_mW", "upper_95_mW"]
            ].rename(columns={
                "forecast_pv_power_mW": "Forecast (mW)",
                "lower_95_mW": "Lower 95% (mW)",
                "upper_95_mW": "Upper 95% (mW)",
            })
            st.line_chart(chart, height=420)

            display_df = forecast.copy()
            display_df["datetime"] = pd.to_datetime(display_df["datetime"]).dt.strftime("%Y-%m-%d %H:%M")
            display_df["forecast_pv_power_mW"] = display_df["forecast_pv_power_mW"].round(1)
            display_df["lower_95_mW"] = display_df["lower_95_mW"].round(1)
            display_df["upper_95_mW"] = display_df["upper_95_mW"].round(1)
            display_df["uncertainty_mW"] = display_df["uncertainty_mW"].round(1)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download forecast CSV",
                data=forecast.to_csv(index=False).encode("utf-8"),
                file_name="live_solar_pv_forecast.csv",
                mime="text/csv",
            )

        with st.expander("Latest sensor values"):
            sensor_view = latest.to_frame("value")
            st.dataframe(sensor_view, use_container_width=True)

        with st.expander("Model information"):
            st.write("Selected models:", SELECTED)
            st.write("Dynamic stacker features:", getattr(stacker, "n_features_in_", "unknown"))
            st.write("95% PI calibration factor:", CAL_FACTOR)
            st.dataframe(
                pd.DataFrame({
                    "Model": SELECTED,
                    "QScore": [q_scores.get(m, np.nan) for m in SELECTED],
                    "Median validation uncertainty (mW)": [
                        assessment.loc[m, "Median_Uncertainty"] if m in assessment.index else np.nan
                        for m in SELECTED
                    ],
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "The app reads the latest Google Sheet history, "
            "recreates the training feature pipeline, runs the saved selected models, "
            "and applies the saved dynamic stacker recursively. "
            "All rights reserved. "
            "Developed by- S. M. Rezaul Karim, PhD Candidate, DUET, "
            "Email: rezaiubat@gmail.com, Cell: +8801725833289"
        )
    live_forecast_panel()

if __name__ == "__main__":
    main()
