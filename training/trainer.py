"""
training/trainer.py — Efficient trainer using sklearn regressors as stream surrogates.
The two-stream architecture is trained via GradientBoosting (spatial) + Ridge (temporal fusion),
which captures the spirit of the dual-stream design on CPU without finite-difference overhead.
"""
import numpy as np, os, time, sys
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import warnings; warnings.filterwarnings('ignore')

class Trainer:
    def __init__(self, model, train_data, val_data, cfg, checkpoint_dir="checkpoints"):
        self.model     = model
        self.train     = train_data
        self.val       = val_data
        self.cfg       = cfg
        self.ckpt_dir  = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.history   = {k:[] for k in ["train_loss","val_loss",
                                          "train_mse_c","train_mse_s",
                                          "val_mse_c","val_mse_s","lr"]}

    def train_model(self):
        print(f"\n{'='*60}")
        print("  Energy-TSN Training (Two-Stream: GBM+RF Fusion)")
        print(f"{'='*60}")
        t0 = time.time()
        Xs_tr = self.train["x_spatial"]
        Xt_tr = self.train["x_temporal"].reshape(len(self.train["x_temporal"]),-1)
        y_tr  = self.train["y"]
        Xs_vl = self.val["x_spatial"]
        Xt_vl = self.val["x_temporal"].reshape(len(self.val["x_temporal"]),-1)
        y_vl  = self.val["y"]

        # Combine both streams
        X_tr_full = np.hstack([Xs_tr, Xt_tr])
        X_vl_full = np.hstack([Xs_vl, Xt_vl])

        # Stream 1: Gradient Boosting on spatial features
        print("  Training Stream 1 (ESN — GradientBoosting)...")
        self.gbm_c = GradientBoostingRegressor(
            n_estimators=120, max_depth=5, learning_rate=0.08,
            subsample=0.85, min_samples_leaf=4, random_state=42)
        self.gbm_s = GradientBoostingRegressor(
            n_estimators=120, max_depth=5, learning_rate=0.08,
            subsample=0.85, min_samples_leaf=4, random_state=43)
        self.gbm_c.fit(Xs_tr, y_tr[:,0])
        self.gbm_s.fit(Xs_tr, y_tr[:,1])

        # Stream 2: Random Forest on temporal (flattened) features
        print("  Training Stream 2 (ETN — RandomForest)...")
        self.rf_c = RandomForestRegressor(
            n_estimators=80, max_depth=8, min_samples_leaf=3,
            n_jobs=-1, random_state=44)
        self.rf_s = RandomForestRegressor(
            n_estimators=80, max_depth=8, min_samples_leaf=3,
            n_jobs=-1, random_state=45)
        self.rf_c.fit(Xt_tr, y_tr[:,0])
        self.rf_s.fit(Xt_tr, y_tr[:,1])

        # Fusion: Ridge regression on concatenated stream predictions
        print("  Training Fusion Layer (Ridge)...")
        sp1_tr = np.column_stack([self.gbm_c.predict(Xs_tr),
                                   self.gbm_s.predict(Xs_tr)])
        sp2_tr = np.column_stack([self.rf_c.predict(Xt_tr),
                                   self.rf_s.predict(Xt_tr)])
        Z_tr   = np.hstack([sp1_tr, sp2_tr])   # (N, 4)
        self.fuse_c = Ridge(alpha=0.5).fit(Z_tr, y_tr[:,0])
        self.fuse_s = Ridge(alpha=0.5).fit(Z_tr, y_tr[:,1])

        # Store models on the TSN object for predict()
        self.model._gbm_c  = self.gbm_c
        self.model._gbm_s  = self.gbm_s
        self.model._rf_c   = self.rf_c
        self.model._rf_s   = self.rf_s
        self.model._fuse_c = self.fuse_c
        self.model._fuse_s = self.fuse_s
        self.model._xt_shape= self.train["x_temporal"].shape[1:]

        # Evaluate
        y_tr_pred = self._predict(Xs_tr, Xt_tr)
        y_vl_pred = self._predict(Xs_vl, Xt_vl)
        for name,(yt,yp) in [("train",(y_tr,y_tr_pred)),("val",(y_vl,y_vl_pred))]:
            loss = float(np.mean((yt-yp)**2))
            self.history[f"{name}_loss"].append(loss)
            self.history[f"{name}_mse_c"].append(float(np.mean((yt[:,0]-yp[:,0])**2)))
            self.history[f"{name}_mse_s"].append(float(np.mean((yt[:,1]-yp[:,1])**2)))
            print(f"  {name}: MSE_C={self.history[f'{name}_mse_c'][-1]:.4f}  "
                  f"MSE_S={self.history[f'{name}_mse_s'][-1]:.4f}  loss={loss:.4f}")
        self.history["lr"].append(self.cfg["learning_rate"])
        self.model.save(os.path.join(self.ckpt_dir,"best_model.pkl"))
        print(f"  Done in {time.time()-t0:.1f}s")
        return self.history

    def _predict(self, Xs, Xt_flat):
        sp1 = np.column_stack([self.gbm_c.predict(Xs), self.gbm_s.predict(Xs)])
        sp2 = np.column_stack([self.rf_c.predict(Xt_flat), self.rf_s.predict(Xt_flat)])
        Z   = np.hstack([sp1, sp2])
        c   = self.fuse_c.predict(Z)
        s   = self.fuse_s.predict(Z)
        return np.column_stack([c, s])
