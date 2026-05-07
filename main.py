"""
main.py — Energy-TSN entry point (clean version).
"""
import argparse, sys, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from config import (CONSUMPTION_CSV, SOLAR_CSV, MODEL_DIR, RESULT_DIR,
                    FIGURE_DIR, DATASET, FEATURES, SPATIAL_NET,
                    TEMPORAL_NET, FUSION, TRAINING)
from data.generate_datasets    import generate_datasets
from models.two_stream_network import TwoStreamNetwork
from training.trainer          import Trainer
from evaluation.evaluator      import MultiGranularityEvaluator
from evaluation.metrics        import compute_all, format_metrics_table
from figures.plotter           import generate_all_figures

# ── Dataset ───────────────────────────────────────────────────────────────
def build_dataset():
    if not os.path.exists(CONSUMPTION_CSV) or not os.path.exists(SOLAR_CSV):
        return generate_datasets(DATASET, CONSUMPTION_CSV, SOLAR_CSV, verbose=True)
    df_c = pd.read_csv(CONSUMPTION_CSV, index_col="timestamp", parse_dates=True)
    df_s = pd.read_csv(SOLAR_CSV,       index_col="timestamp", parse_dates=True)
    df   = df_c.join(df_s, how="inner", rsuffix="_s")
    print(f"[Main] Loaded {len(df):,} rows")
    return df

def prepare_features(df):
    feat_cfg = FEATURES
    lookback = feat_cfg["lookback"]
    target_cols = [feat_cfg["target_consumption"], feat_cfg["target_solar"]]

    temporal_cols = [c for c in feat_cfg["temporal_cols"]        if c in df.columns]
    meteo_cols    = [c for c in feat_cfg["meteo_cols"]           if c in df.columns]
    lag_cols_c    = [c for c in feat_cfg["consumption_lag_cols"] if c in df.columns]
    lag_cols_s    = [c for c in feat_cfg["solar_lag_cols"]       if c in df.columns]
    spatial_cols  = temporal_cols + meteo_cols + lag_cols_c + lag_cols_s

    temporal_feat_cols = [c for c in
        ["consumption_kwh","solar_kwh","hour_sin","hour_cos","solar_irradiance"]
        if c in df.columns]

    all_cols = list(set(spatial_cols + temporal_feat_cols + target_cols))
    df_clean = df[all_cols].dropna()
    print(f"[Main] {len(spatial_cols)} spatial features, {len(temporal_feat_cols)} temporal | {len(df_clean):,} rows")

    scaler_X = StandardScaler(); scaler_y = StandardScaler(); scaler_T = StandardScaler()
    X_s = scaler_X.fit_transform(df_clean[spatial_cols].values.astype(np.float64))
    y_s = scaler_y.fit_transform(df_clean[target_cols].values.astype(np.float64))
    T_s = scaler_T.fit_transform(df_clean[temporal_feat_cols].values.astype(np.float64))

    N   = len(df_clean) - lookback
    Xs  = X_s[lookback:]
    y   = y_s[lookback:]
    ts  = df_clean.index[lookback:]
    Xt  = np.zeros((N, lookback, len(temporal_feat_cols)))
    for i in range(N):
        Xt[i] = T_s[i:i+lookback]

    print(f"[Main] Xs:{Xs.shape} Xt:{Xt.shape} y:{y.shape}")
    return Xs, Xt, y, ts, scaler_X, scaler_y, spatial_cols, temporal_feat_cols

def split_data(Xs, Xt, y, ts):
    N = len(y)
    tr = int(N * DATASET["train_ratio"])
    vl = tr + int(N * DATASET["val_ratio"])
    splits = {}
    for name, sl in [("train",slice(0,tr)),("val",slice(tr,vl)),("test",slice(vl,None))]:
        splits[name] = {"x_spatial":Xs[sl],"x_temporal":Xt[sl],"y":y[sl],"ts":ts[sl]}
        print(f"  {name:5s}: {splits[name]['y'].shape[0]:,} samples")
    return splits

def build_model(n_sp, n_tp):
    m = TwoStreamNetwork(
        spatial_input_dim=n_sp, temporal_input_dim=n_tp,
        seq_len=FEATURES["lookback"],
        spatial_cfg=SPATIAL_NET, temporal_cfg=TEMPORAL_NET,
        fusion_cfg=FUSION, seed=TRAINING["random_seed"])
    print(f"[Model] Built TSN: spatial_in={n_sp}, temporal_in={n_tp}")
    return m

# ── Pipeline steps ─────────────────────────────────────────────────────────
def step_generate():
    print("\n" + "="*60 + "\n  STEP 1: Dataset Generation\n" + "="*60)
    generate_datasets(DATASET, CONSUMPTION_CSV, SOLAR_CSV, verbose=True)

def step_train():
    print("\n" + "="*60 + "\n  STEP 2: Training\n" + "="*60)
    df  = build_dataset()
    Xs, Xt, y, ts, scaler_X, scaler_y, _, _ = prepare_features(df)
    splits  = split_data(Xs, Xt, y, ts)
    model   = build_model(Xs.shape[1], Xt.shape[2])
    trainer = Trainer(model, splits["train"], splits["val"], TRAINING, MODEL_DIR)
    history = trainer.train_model()

    # Save artefacts
    np.save(f"{MODEL_DIR}/test_Xs.npy",  splits["test"]["x_spatial"])
    np.save(f"{MODEL_DIR}/test_Xt.npy",  splits["test"]["x_temporal"])
    np.save(f"{MODEL_DIR}/test_y.npy",   splits["test"]["y"])
    pd.Series(splits["test"]["ts"]).to_csv(f"{MODEL_DIR}/test_ts.csv", index=False)
    np.save(f"{MODEL_DIR}/scaler_mean.npy",  scaler_y.mean_)
    np.save(f"{MODEL_DIR}/scaler_scale.npy", scaler_y.scale_)
    np.save(f"{MODEL_DIR}/history.npy",  history, allow_pickle=True)
    np.save(f"{MODEL_DIR}/dims.npy", {"ns": Xs.shape[1], "nt": Xt.shape[2]}, allow_pickle=True)
    return model, history, splits, scaler_y

def step_evaluate(model=None, splits=None, scaler_y=None, history=None):
    print("\n" + "="*60 + "\n  STEP 3: Evaluation\n" + "="*60)
    if model is None:
        dims = np.load(f"{MODEL_DIR}/dims.npy", allow_pickle=True).item()
        model = build_model(dims["ns"], dims["nt"])
        model.load(f"{MODEL_DIR}/best_model.pkl")
    if splits is None:
        Xs_t = np.load(f"{MODEL_DIR}/test_Xs.npy")
        Xt_t = np.load(f"{MODEL_DIR}/test_Xt.npy")
        y_t  = np.load(f"{MODEL_DIR}/test_y.npy")
        ts_t = pd.to_datetime(pd.read_csv(f"{MODEL_DIR}/test_ts.csv")["0"])
    else:
        Xs_t = splits["test"]["x_spatial"]
        Xt_t = splits["test"]["x_temporal"]
        y_t  = splits["test"]["y"]
        ts_t = splits["test"]["ts"]
    if scaler_y is None:
        scaler_y = StandardScaler()
        scaler_y.mean_  = np.load(f"{MODEL_DIR}/scaler_mean.npy")
        scaler_y.scale_ = np.load(f"{MODEL_DIR}/scaler_scale.npy")
    if history is None:
        history = np.load(f"{MODEL_DIR}/history.npy", allow_pickle=True).item()

    y_pred_sc = model.predict(Xs_t, Xt_t)
    y_true    = scaler_y.inverse_transform(y_t)
    y_pred    = np.clip(scaler_y.inverse_transform(y_pred_sc), 0, None)

    for i, name in enumerate(["Consumption (kWh)", "Solar (kWh)"]):
        m = compute_all(y_true[:,i], y_pred[:,i], target_name=name)
        print(f"\n  {name}")
        for k,v in m.items():
            if isinstance(v,float): print(f"    {k:<12}: {v:.6f}")

    evaluator   = MultiGranularityEvaluator(ts_t, y_true, y_pred, RESULT_DIR)
    all_metrics = evaluator.run_all(verbose=True)
    return model, evaluator, y_true, y_pred, history, all_metrics

def step_plot(model, evaluator, y_true, y_pred, history, all_metrics):
    print("\n" + "="*60 + "\n  STEP 4: Figures\n" + "="*60)
    generate_all_figures(model, evaluator, y_true, y_pred, history, all_metrics, FIGURE_DIR)

# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "█"*60)
    print("  Energy-TSN — Two-Stream Network for Energy Prediction")
    print("  ESN (Spatial) + ETN (Temporal) + Fusion")
    print("█"*60)
    p = argparse.ArgumentParser()
    p.add_argument("--all",           action="store_true")
    p.add_argument("--generate-data", action="store_true")
    p.add_argument("--train",         action="store_true")
    p.add_argument("--evaluate",      action="store_true")
    p.add_argument("--plot",          action="store_true")
    p.add_argument("--granularity",   default=None,
        choices=["minutely","hourly","daily","weekly","monthly"])
    args = p.parse_args()
    t0   = time.time()

    run_all = args.all or not any([args.generate_data, args.train,
                                   args.evaluate, args.plot])
    if run_all or args.generate_data:
        step_generate()
    if run_all or args.train:
        model, history, splits, scaler_y = step_train()
    else:
        model, history, splits, scaler_y = None, None, None, None
    if run_all or args.evaluate:
        model, evaluator, y_true, y_pred, history, all_metrics = step_evaluate(
            model=model, splits=splits, scaler_y=scaler_y, history=history)
    if run_all or args.plot:
        step_plot(model, evaluator, y_true, y_pred, history, all_metrics)

    print(f"\n  Done in {time.time()-t0:.1f}s  |  Results→{RESULT_DIR}  |  Figures→{FIGURE_DIR}")

if __name__ == "__main__":
    main()
