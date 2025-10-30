import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
# Import model-model lain jika Anda ingin melatihnya juga
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
import sys
import warnings
import joblib
import os
import yaml
import sklearn
import shutil
import dagshub  # <-- DITAMBAHKAN
import matplotlib.pyplot as plt # <-- DITAMBAHKAN
import numpy as np # <-- DITAMBAHKAN

# --- Inisialisasi DagsHub ---
dagshub.init(repo_owner='bisat19',
             repo_name='Membangun_model',
             mlflow=True)
# -----------------------------

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Memuat data dari path CSV dengan error handling lebih baik."""
    print(f"Mencoba membaca file dari: {file_path}") # Verifikasi path
    try:
        if not os.path.exists(file_path):
            print(f"Error: File tidak ditemukan di path: {file_path}", file=sys.stderr)
            return None
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"Error: File CSV kosong: {file_path}", file=sys.stderr)
            return None
        print("File CSV berhasil dibaca.")
        return df
    except pd.errors.EmptyDataError:
        print(f"Error EmptyDataError: File CSV kosong: {file_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error saat membaca CSV ({type(e).__name__}): {e}", file=sys.stderr)
        return None

# --- FUNGSI EVAL_METRICS LENGKAP ---
def eval_metrics(y_test, y_pred, y_pred_proba):
    """Menghitung metrik (termasuk advance) dan mengembalikannya sebagai dictionary."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        y_pred_proba_clipped = np.clip(y_pred_proba, 1e-6, 1 - 1e-6)
        ll = log_loss(y_test, y_pred_proba_clipped)
    except ValueError as e:
        print(f"Warning: Gagal menghitung log_loss: {e}. Mengatur ke -1.")
        ll = -1
    try:
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        elif y_pred_proba.ndim == 1:
             auc = roc_auc_score(y_test, y_pred_proba)
        else:
             print("Warning: Tidak bisa menghitung AUC, format probabilitas tidak dikenali.")
             auc = -1
    except ValueError as e:
        print(f"Warning: Gagal menghitung AUC: {e}. Mengatur ke -1.")
        auc = -1

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "log_loss": ll, "roc_auc": auc}
# ------------------------------------

# --- FUNGSI CONFUSION MATRIX ---
def log_confusion_matrix(y_test, y_pred, run_name):
    """Membuat, menyimpan, dan log plot confusion matrix."""
    try:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {run_name}')
        cm_filename = f"{run_name}_confusion_matrix.png"
        plt.savefig(cm_filename)
        plt.close()
        mlflow.log_artifact(cm_filename, artifact_path="evaluation_plots")
        print(f"Confusion matrix disimpan dan di-log.")
        os.remove(cm_filename)
    except Exception as e:
        print(f"Gagal membuat atau log confusion matrix: {e}")
# -------------------------------

def main():
    # --- Path yang Lebih Robust ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(script_dir, "PCOS_preprocessing.csv")
    SERVING_ENV_PATH_SOURCE = os.path.join(script_dir, "..", "model_env.yaml")
    # ---------------------------------
    SERVING_ENV_PATH_DEST = "conda.yaml"

    print("Memuat data...")
    data = load_data(DATA_PATH)

    # --- Tambahkan Pengecekan data is None ---
    if data is None:
        print("Gagal memuat data. Menghentikan script.", file=sys.stderr)
        sys.exit(1)
    # -----------------------------------------

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

    # --- Ganti Nama Run (Opsional) ---
    run_name = "RF_CI_Training_Run_Manual"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Memulai run: {run.info.run_id}")

        print("Melatih model...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) # <-- DITAMBAHKAN

        # --- LOGGING METRIK LENGKAP ---
        print("Logging parameter dan metrik lengkap...")
        metrics = eval_metrics(y_test, y_pred, y_pred_proba)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics) # <-- Diubah ke metrics (plural)
        print(f"  Metrics: {metrics}")
        # ---------------------------------

        # --- LOGGING CONFUSION MATRIX ---
        print("Logging confusion matrix...")
        log_confusion_matrix(y_test, y_pred, run_name)
        # ---------------------------------

        print("Membuat artefak model (MLmodel & .pkl) secara manual...")
        model_artifact_dir = "model_artifact_temp"
        if not os.path.exists(model_artifact_dir):
            os.makedirs(model_artifact_dir)

        model_path = os.path.join(model_artifact_dir, "model.pkl")
        joblib.dump(model, model_path)

        sklearn_version = sklearn.__version__
        conda_env_path = "conda.yaml"
        python_version = "3.10"

        # --- PERBAIKAN SIGNATURE (untuk Kriteria 4) ---
        try:
           signature_obj = mlflow.models.infer_signature(X_train, model.predict(X_train))
           signature_dict = signature_obj.to_dict() # Ubah ke dict
        except Exception as e:
           print(f"Warning: Gagal menginfer signature model: {e}")
           signature_dict = None
        # ------------------------------------------

        mlmodel_dict = {
            'flavors': {
                'python_function': {
                    'env': {'conda': conda_env_path},
                    'loader_module': 'mlflow.sklearn',
                    'model_path': 'model.pkl',
                    'python_version': python_version
                },
                'sklearn': {
                    'pickled_model': 'model.pkl',
                    'serialization_format': 'cloudpickle',
                    'sklearn_version': sklearn_version
                }
            },
            'run_id': run.info.run_id,
            'signature': signature_dict, # <-- Gunakan dict
            'utc_time_created': pd.Timestamp.utcnow().isoformat() + "Z"
        }

        mlmodel_path = os.path.join(model_artifact_dir, "MLmodel")
        with open(mlmodel_path, 'w') as f:
            yaml.dump(mlmodel_dict, f, default_flow_style=False)

        print(f"Menyalin {SERVING_ENV_PATH_SOURCE} ke {os.path.join(model_artifact_dir, SERVING_ENV_PATH_DEST)}...")
        # Perbaiki path copy
        source_env_file = SERVING_ENV_PATH_SOURCE
        dest_env_file = os.path.join(model_artifact_dir, conda_env_path)

        if not os.path.exists(source_env_file):
             print(f"Error: File environment sumber {source_env_file} tidak ditemukan! File conda.yaml tidak akan ditambahkan.", file=sys.stderr)
        else:
            try:
                shutil.copyfile(source_env_file, dest_env_file)
            except Exception as e:
                print(f"Error saat menyalin file environment: {e}", file=sys.stderr)


        mlflow.log_artifacts(model_artifact_dir, artifact_path="model")
        print("Artefak model manual berhasil di-log ke 'model'.")
        
        # Hapus folder sementara
        shutil.rmtree(model_artifact_dir)

        run_id = run.info.run_id
        # Simpan run_id di folder root (satu level di atas)
        with open("../run_id.txt", "w") as f:
            f.write(run_id)
        print(f"Menyimpan Run ID: {run_id}")

if __name__ == "__main__":
    main()
