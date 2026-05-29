"""
Reservoir Proxy Model — Inference Script
=========================================
Predicts 22-year oil production curves and scalar reservoir targets
from six input parameters using the trained models from this project.

Usage
-----
    python predict.py --model lstm --perm 1.5 --poro 1.2 --bhp 2000 \
                      --inj_rate 500 --inj_bhp 8000 --init_period 365

    python predict.py --model all --perm 1.0 --poro 1.0 --bhp 3000 \
                      --inj_rate 35 --inj_bhp 10000 --init_period 3230 --plot

Models
------
    lstm   — Encoder-Decoder LSTM: predicts full 22-year time-series for
             8 physical outputs (oil rate, cumulative oil, pressure, gas, etc.)
    mlp    — MLP scalar proxy: predicts 4 final/peak scalar values instantly
    pinn   — Physics-Informed NN: same 4 scalars with physics constraints enforced
    rf     — Random Forest baseline: same 4 scalars, most interpretable
    all    — Run all four models and print a comparison table

Inputs (all required)
---------------------
    --perm          Permeability multiplier         (e.g. 0.1 – 10.0)
    --poro          Porosity multiplier             (e.g. 0.5 – 2.0)
    --bhp           Producer BHP pressure (psi)    (e.g. 1000 – 5000)
    --inj_rate      Gas injection rate (Mscf/d)    (e.g. 5 – 100)
    --inj_bhp       Injector BHP limit (psi)        (e.g. 5000 – 15000)
    --init_period   Initial production period (days)(e.g. 100 – 5000)

Outputs
-------
    LSTM  : full time-series DataFrame (days × 8 variables)
    Others: scalar predictions printed to console
    --plot: optional matplotlib figure saved as prediction_plot.png
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd

SAVED = "saved_models"
FEATURES = [
    "producer_bhp_psi", "gas_inj_rate_mscf_d", "inj_bhp_limit_psi",
    "init_prod_period_days", "perm_multiplier", "poro_multiplier",
]
SCALAR_TARGETS = ["final_fopt", "final_fpr", "final_fopr", "peak_fopr"]
SCALAR_UNITS   = {"final_fopt": "STB", "final_fpr": "psi",
                  "final_fopr": "STB/d", "peak_fopr": "STB/d"}


def load_scenario(args):
    return pd.DataFrame([{
        "producer_bhp_psi":      args.bhp,
        "gas_inj_rate_mscf_d":   args.inj_rate,
        "inj_bhp_limit_psi":     args.inj_bhp,
        "init_prod_period_days": args.init_period,
        "perm_multiplier":       args.perm,
        "poro_multiplier":       args.poro,
    }])


def predict_lstm(scenario, plot=False):
    import joblib, tensorflow as tf

    model = tf.keras.models.load_model(f"{SAVED}/enc_dec_lstm_reservoir_proxy.keras",
                                       compile=False)
    scaler_static = joblib.load(f"{SAVED}/scaler_static.pkl")
    scaler_time   = joblib.load(f"{SAVED}/scaler_time.pkl")
    scaler_y      = joblib.load(f"{SAVED}/scaler_y.pkl")
    avg_time_grid = np.load(f"{SAVED}/lstm_avg_time_grid.npy")
    with open(f"{SAVED}/lstm_metadata.json") as f:
        meta = json.load(f)

    MAX_LEN        = meta["MAX_LEN"]
    OUTPUT_TARGETS = meta["OUTPUT_TARGETS"]
    TARGET_UNITS   = meta["TARGET_UNITS"]
    ENC_FEATURES   = meta["ENCODER_FEATURES"]

    X_static = scaler_static.transform(scenario[ENC_FEATURES].values)
    X_time   = scaler_time.transform(avg_time_grid.reshape(-1, 1)).reshape(1, MAX_LEN, 1)

    y_s   = model.predict([X_static, X_time], verbose=0)
    y_out = scaler_y.inverse_transform(y_s.reshape(-1, len(OUTPUT_TARGETS)))
    times = avg_time_grid

    df_out = pd.DataFrame(y_out, columns=OUTPUT_TARGETS)
    df_out.insert(0, "time_days", times)

    print("\n=== LSTM  —  22-year production forecast ===")
    print(f"{'Variable':<12}  {'Final value':>15}  Unit")
    print("-" * 40)
    for col, unit in zip(OUTPUT_TARGETS, TARGET_UNITS):
        print(f"{col:<12}  {df_out[col].iloc[-1]:>15,.2f}  {unit}")

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("LSTM Proxy — 22-Year Forecast", fontsize=13, fontweight="bold")
        for ax, col, color, unit in zip(
            axes,
            ["fopr", "fopt", "fpr"],
            ["#e74c3c", "#2ecc71", "#3498db"],
            ["STB/day", "STB", "PSIA"],
        ):
            ax.plot(df_out["time_days"], df_out[col], color=color, linewidth=2)
            ax.set_title(col.upper()); ax.set_xlabel("Days"); ax.set_ylabel(unit)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("prediction_plot.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to prediction_plot.png")

    return df_out


def predict_mlp(scenario):
    import joblib, tensorflow as tf

    model    = tf.keras.models.load_model(f"{SAVED}/mlp_reservoir_proxy.keras", compile=False)
    scaler_X = joblib.load(f"{SAVED}/mlp_scaler_X.pkl")
    scaler_y = joblib.load(f"{SAVED}/mlp_scaler_y.pkl")

    X_s  = scaler_X.transform(scenario[FEATURES].values)
    y_s  = model.predict(X_s, verbose=0)
    y    = scaler_y.inverse_transform(y_s)[0]

    print("\n=== MLP  —  Scalar predictions ===")
    for name, val in zip(SCALAR_TARGETS, y):
        print(f"  {name:<18} {val:>12,.2f}  {SCALAR_UNITS[name]}")
    return dict(zip(SCALAR_TARGETS, y))


def predict_pinn(scenario):
    import joblib, tensorflow as tf

    model    = tf.keras.models.load_model(f"{SAVED}/pinn_base_model.keras", compile=False)
    scaler_X = joblib.load(f"{SAVED}/pinn_scaler_X.pkl")
    scaler_y = joblib.load(f"{SAVED}/pinn_scaler_y.pkl")

    X_s  = scaler_X.transform(scenario[FEATURES].values)
    y_s  = model.predict(X_s, verbose=0)
    y    = scaler_y.inverse_transform(y_s)[0]

    print("\n=== PINN  —  Scalar predictions (physics-constrained) ===")
    for name, val in zip(SCALAR_TARGETS, y):
        print(f"  {name:<18} {val:>12,.2f}  {SCALAR_UNITS[name]}")
    return dict(zip(SCALAR_TARGETS, y))


def predict_rf(scenario):
    import joblib

    print("\n=== Random Forest  —  Scalar predictions ===")
    results = {}
    for key in SCALAR_TARGETS:
        model = joblib.load(f"{SAVED}/rf_{key}.pkl")
        val   = model.predict(scenario[FEATURES].values)[0]
        results[key] = val
        print(f"  {key:<18} {val:>12,.2f}  {SCALAR_UNITS[key]}")
    return results


def compare_all(scenario, plot=False):
    mlp  = predict_mlp(scenario)
    pinn = predict_pinn(scenario)
    rf   = predict_rf(scenario)

    print("\n=== Model Comparison  —  Scalar targets ===")
    print(f"{'Target':<18}  {'MLP':>12}  {'PINN':>12}  {'RF':>12}  Unit")
    print("-" * 65)
    for key in SCALAR_TARGETS:
        print(f"{key:<18}  {mlp[key]:>12,.1f}  {pinn[key]:>12,.1f}  "
              f"{rf[key]:>12,.1f}  {SCALAR_UNITS[key]}")

    predict_lstm(scenario, plot=plot)


def main():
    parser = argparse.ArgumentParser(
        description="Reservoir proxy model — run inference with trained models"
    )
    parser.add_argument("--model", choices=["lstm", "mlp", "pinn", "rf", "all"],
                        default="lstm", help="Which model to use (default: lstm)")
    parser.add_argument("--perm",        type=float, required=True,
                        help="Permeability multiplier")
    parser.add_argument("--poro",        type=float, required=True,
                        help="Porosity multiplier")
    parser.add_argument("--bhp",         type=float, required=True,
                        help="Producer BHP (psi)")
    parser.add_argument("--inj_rate",    type=float, required=True,
                        help="Gas injection rate (Mscf/d)")
    parser.add_argument("--inj_bhp",     type=float, required=True,
                        help="Injector BHP limit (psi)")
    parser.add_argument("--init_period", type=float, required=True,
                        help="Initial production period (days)")
    parser.add_argument("--plot", action="store_true",
                        help="Save a prediction plot as prediction_plot.png")

    args = parser.parse_args()
    scenario = load_scenario(args)

    dispatch = {
        "lstm": lambda: predict_lstm(scenario, plot=args.plot),
        "mlp":  lambda: predict_mlp(scenario),
        "pinn": lambda: predict_pinn(scenario),
        "rf":   lambda: predict_rf(scenario),
        "all":  lambda: compare_all(scenario, plot=args.plot),
    }
    dispatch[args.model]()


if __name__ == "__main__":
    main()
