"""
Download pre-trained model weights from HuggingFace Hub.

Usage
-----
    pip install huggingface_hub
    python download_models.py

This creates the `saved_models/` directory and downloads all weights,
scalers, and metadata needed to run predict.py and demo.ipynb without
having to retrain the models.
"""

import os
from huggingface_hub import hf_hub_download

HF_REPO  = "mushahid-raza/reservoir-proxy-models"   # ← update if your HF username differs
OUT_DIR  = "saved_models"

FILES = [
    # LSTM
    "enc_dec_lstm_reservoir_proxy.keras",
    "scaler_static.pkl",
    "scaler_time.pkl",
    "scaler_y.pkl",
    "lstm_avg_time_grid.npy",
    "lstm_metadata.json",
    # MLP
    "mlp_reservoir_proxy.keras",
    "mlp_scaler_X.pkl",
    "mlp_scaler_y.pkl",
    "mlp_metadata.json",
    # PINN
    "pinn_base_model.keras",
    "pinn_scaler_X.pkl",
    "pinn_scaler_y.pkl",
    "pinn_metadata.json",
    # Random Forest
    "rf_final_fopt.pkl",
    "rf_final_fpr.pkl",
    "rf_final_fopr.pkl",
    "rf_peak_fopr.pkl",
    "rf_metadata.json",
]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {len(FILES)} files from {HF_REPO} ...\n")

    for fname in FILES:
        dest = os.path.join(OUT_DIR, fname)
        if os.path.exists(dest):
            print(f"  [skip]  {fname}  (already exists)")
            continue
        print(f"  ↓  {fname}")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=fname,
            local_dir=OUT_DIR,
        )

    print(f"\nAll files saved to {OUT_DIR}/")
    print("You can now run:  python predict.py --model all --perm 1.0 --poro 1.0 "
          "--bhp 3000 --inj_rate 35 --inj_bhp 10000 --init_period 3230")

if __name__ == "__main__":
    main()
