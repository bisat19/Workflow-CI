import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
import warnings
import joblib
import os
import yaml 
import sklearn 

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Memuat data dari path yang benar."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}", file=sys.stderr)
        sys.exit(1)

def main():
    DATA_PATH = "../MLProject/PCOS_preprocessing.csv"
    
    print("Memuat data...")
    data = load_data(DATA_PATH)
    
    try:
        X = data.drop('PCOS (Y/N)', axis=1)
        y = data['PCOS (Y/N)']
    except KeyError as e:
        print(f"Error: Kolom target 'PCOS (Y/N)' tidak ditemukan. {e}", file=sys.stderr)
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        "n_estimators": 100,
        "max_depth": None, 
        "random_state": 42
    }
    model = RandomForestClassifier(**params)

    with mlflow.start_run(run_name="CI_Training_Run_Manual") as run:
        print(f"Memulai run: {run.info.run_id}")
        
        print("Melatih model...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Accuracy: {accuracy:.4f}")

        print("Logging parameter dan metrik...")
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        print("Membuat artefak model (MLmodel & .pkl) secara manual...")

        # 1. Tentukan direktori sementara untuk artefak
        model_artifact_dir = "model_artifact_temp"
        if not os.path.exists(model_artifact_dir):
            os.makedirs(model_artifact_dir)

        # 2. Simpan model.pkl di dalam direktori
        model_path = os.path.join(model_artifact_dir, "model.pkl")
        joblib.dump(model, model_path)

        # 3. Buat file MLmodel secara manual
        sklearn_version = sklearn.__version__
        conda_env_path = "../conda.yaml"
        python_version = "3.12.1" 

        # Konten untuk file MLmodel
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
            'utc_time_created': pd.Timestamp.utcnow().isoformat() + "Z"
        }
        
        # Tulis file MLmodel ke direktori
        mlmodel_path = os.path.join(model_artifact_dir, "MLmodel")
        with open(mlmodel_path, 'w') as f:
            yaml.dump(mlmodel_dict, f, default_flow_style=False)

        # 4. Log SELURUH DIREKTORI sebagai artefak
        mlflow.log_artifacts(model_artifact_dir, artifact_path="model")

        print("Artefak model manual berhasil di-log ke 'model'.")

        run_id = run.info.run_id
        with open("../run_id.txt", "w") as f:
            f.write(run_id)
        print(f"Menyimpan Run ID: {run_id}")

if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("Error: 'pyyaml' tidak terinstal. Harap tambahkan ke conda.yaml.", file=sys.stderr)
        sys.exit(1)
        
    main()