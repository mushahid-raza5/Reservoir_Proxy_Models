"""
Upload trained model weights to HuggingFace Hub.

Usage
-----
    pip install huggingface_hub
    huggingface-cli login          # paste your HF token when prompted
    python upload_models.py

What it does
------------
    Uploads every file in saved_models/ to your HuggingFace repository so
    that anyone can download them instantly with download_models.py.
"""

import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "mushahid-raza"          # ← your HuggingFace username
REPO_NAME   = "reservoir-proxy-models"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"
SAVED_DIR   = "saved_models"

EXPECTED_FILES = [
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
    # Check saved_models/ exists
    if not os.path.isdir(SAVED_DIR):
        print("ERROR: saved_models/ folder not found.")
        print("Run all 4 training notebooks to completion first.")
        return

    # Check which files are present
    missing = [f for f in EXPECTED_FILES if not os.path.exists(os.path.join(SAVED_DIR, f))]
    if missing:
        print(f"WARNING: {len(missing)} expected file(s) not found:")
        for m in missing:
            print(f"  - {m}")
        print("These notebooks may not have been run yet:")
        print("  ML2_FinalProject_LSTM_Model.ipynb")
        print("  ML2_FinalProject_MLPModel.ipynb")
        print("  ML2_FinalProject_Physics_Informed_NN.ipynb")
        print("  ML2_FinalProject_Random_Forest_Baseline.ipynb")
        cont = input("\nContinue uploading the files that do exist? [y/N] ")
        if cont.strip().lower() != "y":
            return

    api = HfApi()

    # Create the repo if it doesn't exist yet
    print(f"Creating / verifying repo: {REPO_ID}")
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    # Upload all files in saved_models/
    all_files = [f for f in os.listdir(SAVED_DIR)
                 if os.path.isfile(os.path.join(SAVED_DIR, f))]

    print(f"\nUploading {len(all_files)} file(s) to {REPO_ID} ...\n")
    for fname in sorted(all_files):
        local_path = os.path.join(SAVED_DIR, fname)
        size_mb = os.path.getsize(local_path) / 1_048_576
        print(f"  ↑  {fname}  ({size_mb:.1f} MB)")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type="model",
        )

    print(f"\nDone! Models live at:")
    print(f"  https://huggingface.co/{REPO_ID}")
    print(f"\nAnyone can now download them with:")
    print(f"  python download_models.py")


if __name__ == "__main__":
    main()
