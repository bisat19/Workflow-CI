import pandas as pd
import mlflow
import mlflow.sklearn # Diperlukan untuk loader_module di MLmodel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sys
import warnings
import joblib
import os
import yaml
import sklearn
import shutil
import dagshub # Pastikan ini diimpor
import matplotlib.pyplot as plt
import numpy as np

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

def eval_metrics(y_test, y_pred, y_pred_proba):
    """Menghitung metrik."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # --- PERBAIKAN try-except untuk log_loss ---
    try:
        # Pastikan probabilitas ada di rentang [0, 1] sebelum log_loss
        y_pred_proba_clipped = np.clip(y_pred_proba, 1e-6, 1 - 1e-6)
        ll = log_loss(y_test, y_pred_proba_clipped)
    except ValueError as e:
        print(f"Warning: Gagal menghitung log_loss: {e}. Mengatur ke -1.")
        ll = -1
    # ---------------------------------------------

    # --- PERBAIKAN Handling AUC ---
    try:
        # Gunakan probabilitas kelas positif (kolom 1) untuk AUC jika ada
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        # Jika hanya 1D (dari decision_function yang sudah di-scale), gunakan itu
        elif y_pred_proba.ndim == 1:
             auc = roc_auc_score(y_test, y_pred_proba)
        else:
             print("Warning: Tidak bisa menghitung AUC, format probabilitas tidak dikenali.")
             auc = -1
    except ValueError as e:
        print(f"Warning: Gagal menghitung AUC: {e}. Mengatur ke -1.")
        auc = -1
    # ----------------------------

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "log_loss": ll, "roc_auc": auc}

# --- PERBAIKAN: Kembalikan definisi fungsi log_confusion_matrix ---
def log_confusion_matrix(y_test, y_pred, run_name):
    """Membuat, menyimpan, dan log plot confusion matrix."""
    try:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {run_name}')
        cm_filename = f"{run_name}_confusion_matrix.png"
        plt.savefig(cm_filename)
        plt.close() # Tutup plot agar tidak tampil di layar
        mlflow.log_artifact(cm_filename, artifact_path="evaluation_plots")
        print(f"Confusion matrix disimpan dan di-log.")
        os.remove(cm_filename)
    except Exception as e:
        print(f"Gagal membuat atau log confusion matrix: {e}")
# -------------------------------------------------------------

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
    # Handle infinite values if any after loading/preprocessing
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Simple imputation: fill NaN with column mean (consider more sophisticated methods if needed)
    if X.isnull().sum().sum() > 0:
        print("Warning: Menemukan NaN dalam fitur, mengisi dengan rata-rata kolom.")
        X.fillna(X.mean(), inplace=True)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Parameter Lists ---
    rf_params_list = [ {"n_estimators": 100, "max_depth": None, "random_state": 42}, {"n_estimators": 150, "max_depth": None, "random_state": 42}, ]
    lr_params_list = [ {"C": 0.1, "max_iter": 1000, "random_state": 42}, {"C": 1.0, "max_iter": 1000, "random_state": 42}, ]
    svm_params_list = [ {"C": 0.1, "kernel": "linear", "random_state": 42, "probability": True}, {"C": 1.0, "kernel": "rbf", "random_state": 42, "probability": True}, ]

    with mlflow.start_run(run_name="All Model Hyperparameter Tuning") as parent_run:
        print(f"Memulai Parent Run: {parent_run.info.run_id}")

        model_types = { "RF": (RandomForestClassifier, rf_params_list), "LR": (LogisticRegression, lr_params_list), "SVM": (SVC, svm_params_list) }

        for model_prefix, (ModelClass, params_list) in model_types.items():
            print(f"\n--- Memulai Tuning {ModelClass.__name__} ---")
            for i, params in enumerate(params_list):
                run_name = f"{model_prefix}_Run_{i+1}"
                with mlflow.start_run(run_name=run_name, nested=True) as run:
                    print(f"Memulai {run_name} dengan params: {params}")

                    # 1. Latih Model
                    model = ModelClass(**params)
                    try:
                        model.fit(X_train, y_train)
                    except ValueError as e:
                        print(f"ERROR saat melatih {run_name}: {e}. Skipping run ini.")
                        continue # Lanjut ke iterasi berikutnya jika training gagal


                    # 2. Lakukan Prediksi
                    y_pred = model.predict(X_test)
                    if hasattr(model, "predict_proba"):
                        y_pred_proba = model.predict_proba(X_test)
                    else: # Handle decision_function
                        try:
                            y_pred_proba_raw = model.decision_function(X_test)
                            # Scaling decision function scores to pseudo-probabilities [0, 1]
                            if y_pred_proba_raw.ndim == 1: # Binary case
                                min_val = y_pred_proba_raw.min()
                                max_val = y_pred_proba_raw.max()
                                if max_val == min_val:
                                    prob_pos_scaled = np.full_like(y_pred_proba_raw, 0.5)
                                else:
                                    prob_pos_scaled = (y_pred_proba_raw - min_val) / (max_val - min_val)
                                y_pred_proba = np.vstack((1 - prob_pos_scaled, prob_pos_scaled)).T
                            else: # Multiclass case (ambil kolom kelas positif jika relevan)
                                print("Warning: Scaling decision_function multiclass belum diimplementasikan dengan baik.")
                                # Ambil kolom pertama sebagai proxy, mungkin perlu disesuaikan
                                prob_pos = y_pred_proba_raw[:, 0]
                                min_val = prob_pos.min()
                                max_val = prob_pos.max()
                                if max_val == min_val:
                                     prob_pos_scaled = np.full_like(prob_pos, 0.5)
                                else:
                                     prob_pos_scaled = (prob_pos - min_val) / (max_val - min_val)
                                # Ini asumsi kasar untuk biner dari multiclass decision func
                                y_pred_proba = np.vstack((1 - prob_pos_scaled, prob_pos_scaled)).T

                        except Exception as e:
                            print(f"Warning: Gagal mendapatkan/memproses decision_function untuk {run_name}: {e}. Membuat probabilitas dummy.")
                            # Buat proba dummy jika gagal total (misal 0.5 untuk semua)
                            y_pred_proba = np.full((len(y_test), 2), 0.5)


                    # 3. Hitung & Log Metrik (Manual)
                    metrics = eval_metrics(y_test, y_pred, y_pred_proba)
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    print(f"Metrik untuk {run_name}: {metrics}")

                    # 4. Log Confusion Matrix (Manual - Artefak Kustom)
                    log_confusion_matrix(y_test, y_pred, run_name)

                    # --- 5. BUAT & LOG ARTEFAK MODEL MANUAL ---
                    print("Membuat artefak model (MLmodel & .pkl) secara manual...")
                    model_artifact_dir = f"{run_name}_model_artifact_temp"
                    if not os.path.exists(model_artifact_dir):
                        os.makedirs(model_artifact_dir)

                    model_pkl_path = os.path.join(model_artifact_dir, "model.pkl")
                    joblib.dump(model, model_pkl_path)

                    sklearn_version = sklearn.__version__
                    try:
                        import cloudpickle
                        cloudpickle_version = cloudpickle.__version__
                    except ImportError:
                        cloudpickle_version = "Not Found"

                    # Infer signature
                    try:
                       signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
                    except Exception as e:
                       print(f"Warning: Gagal menginfer signature model: {e}")
                       signature = None # Tetap lanjutkan tanpa signature jika gagal

                    mlmodel_dict = {
                        'artifact_path': 'model',
                        'flavors': {
                            'python_function': {
                                'env': {'conda': SERVING_ENV_PATH_DEST},
                                'loader_module': 'mlflow.sklearn',
                                'model_path': 'model.pkl',
                                'python_version': "3.10"
                            },
                            'sklearn': {
                                'pickled_model': 'model.pkl',
                                'serialization_format': 'cloudpickle',
                                'sklearn_version': sklearn_version,
                            }
                        },
                        'run_id': run.info.run_id,
                        'signature': signature, # Gunakan signature yang diinfer atau None
                        'utc_time_created': pd.Timestamp.utcnow().isoformat() + "Z"
                    }
                    mlmodel_path = os.path.join(model_artifact_dir, "MLmodel")
                    with open(mlmodel_path, 'w') as f:
                        yaml.dump(mlmodel_dict, f, default_flow_style=False)

                    print(f"Menyalin {SERVING_ENV_PATH_SOURCE} ke {os.path.join(model_artifact_dir, SERVING_ENV_PATH_DEST)}...")
                    if not os.path.exists(SERVING_ENV_PATH_SOURCE):
                         print(f"Error: File environment sumber {SERVING_ENV_PATH_SOURCE} tidak ditemukan! File conda.yaml tidak akan ditambahkan ke artefak model.", file=sys.stderr)
                    else:
                         try:
                             shutil.copyfile(SERVING_ENV_PATH_SOURCE, os.path.join(model_artifact_dir, SERVING_ENV_PATH_DEST))
                         except Exception as e:
                             print(f"Error saat menyalin file environment: {e}", file=sys.stderr)


                    print(f"Logging direktori artefak '{model_artifact_dir}' ke MLflow path 'model'...")
                    mlflow.log_artifacts(model_artifact_dir, artifact_path="model")

                    print(f"Artefak model manual berhasil di-log ke 'model'.")
                    shutil.rmtree(model_artifact_dir)
                    # --- AKHIR PEMBUATAN MANUAL ---

                    print(f"Selesai {run_name}.")

        print(f"\nSemua eksperimen tuning selesai. Parent Run ID: {parent_run.info.run_id}")

if __name__ == "__main__":
    main()