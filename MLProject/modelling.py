import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
import sys
import warnings
import os
import sklearn
import dagshub
import matplotlib.pyplot as plt
import numpy as np

# --- Inisialisasi DagsHub untuk tracking eksperimen (opsional) ---
# Hanya untuk log parameter & metrik, BUKAN untuk model
dagshub.init(repo_owner='bisat19', repo_name='Membangun_model', mlflow=True)

warnings.filterwarnings('ignore')

def load_data(file_path):
    print(f"Mencoba membaca file dari: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File tidak ditemukan di path: {file_path}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"Error: File CSV kosong: {file_path}", file=sys.stderr)
            return None
        print("File CSV berhasil dibaca.")
        return df
    except Exception as e:
        print(f"Error saat membaca CSV: {e}", file=sys.stderr)
        return None

def eval_metrics(y_test, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        y_pred_proba_clipped = np.clip(y_pred_proba, 1e-6, 1 - 1e-6)
        ll = log_loss(y_test, y_pred_proba_clipped)
    except Exception as e:
        print(f"Warning: Gagal menghitung log_loss: {e}. Mengatur ke -1.")
        ll = -1

    try:
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        elif y_pred_proba.ndim == 1:
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = -1
    except Exception as e:
        print(f"Warning: Gagal menghitung AUC: {e}. Mengatur ke -1.")
        auc = -1

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "log_loss": ll, "roc_auc": auc}

def log_confusion_matrix(y_test, y_pred, run_name):
    try:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {run_name}')
        cm_filename = f"{run_name}_confusion_matrix.png"
        plt.savefig(cm_filename)
        plt.close()
        mlflow.log_artifact(cm_filename, artifact_path="evaluation_plots")
        os.remove(cm_filename)
        print("Confusion matrix disimpan dan di-log.")
    except Exception as e:
        print(f"Gagal membuat atau log confusion matrix: {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(script_dir, "PCOS_preprocessing.csv")
    CONDA_ENV_PATH = os.path.join(script_dir, "..", "model_env.yaml")

    print("Memuat data...")
    data = load_data(DATA_PATH)
    if data is None:
        sys.exit(1)

    try:
        X = data.drop('PCOS (Y/N)', axis=1)
        y = data['PCOS (Y/N)']
    except KeyError as e:
        print(f"Error: Kolom target 'PCOS (Y/N)' tidak ditemukan. {e}", file=sys.stderr)
        sys.exit(1)

    # Handle NaN/Inf
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isnull().sum().sum() > 0:
        print("Warning: Menemukan NaN dalam fitur, mengisi dengan rata-rata kolom.")
        X.fillna(X.mean(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42
    }
    model = RandomForestClassifier(**params)

    run_name = "RF_CI_Training_Run_Manual"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Memulai run: {run.info.run_id}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Log parameter & metrik ke DagsHub
        metrics = eval_metrics(y_test, y_pred, y_pred_proba)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")

        # Log confusion matrix
        log_confusion_matrix(y_test, y_pred, run_name)

        # === SIMPAN MODEL SECARA LOKAL (BUKAN KE DAGSHUB) ===
        # Nonaktifkan tracking URI sementara untuk penyimpanan model
        local_tracking_uri = "./mlruns"  # default lokal
        current_uri = mlflow.get_tracking_uri()

        # Simpan model ke filesystem lokal
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        
        if not os.path.exists(CONDA_ENV_PATH):
            print(f"ERROR: {CONDA_ENV_PATH} tidak ditemukan. Pastikan file ini ada!", file=sys.stderr)
            sys.exit(1)

        # Simpan model ke folder lokal
        saved_model_path = "saved_model"
        mlflow.sklearn.save_model(
            model,
            path="saved_model",  
            conda_env="../model_env.yaml",
            signature=signature
        )
        print(f"Model disimpan secara lokal di: {os.path.abspath(saved_model_path)}")

        # Opsional: log folder model sebagai artefak ke DagsHub (hanya metadata, bukan untuk serve)
        mlflow.log_artifacts(saved_model_path, artifact_path="model_artifacts")

        # Simpan run_id
        with open("../run_id.txt", "w") as f:
            f.write(run.info.run_id)
        print(f"Run ID disimpan: {run.info.run_id}")

if __name__ == "__main__":
    main()