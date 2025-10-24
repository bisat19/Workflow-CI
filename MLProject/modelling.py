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

warnings.filterwarnings('ignore')

dagshub.init(repo_owner='bisat19',
             repo_name='Membangun_model',
             mlflow=True)

def load_data(file_path):
    """Memuat data dari path yang benar."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}", file=sys.stderr)
        sys.exit(1)

def main():
    # Path ini sudah benar (relatif dari folder MLProject)
    DATA_PATH = "../namadataset_preprocessing/PCOS_preprocessing.csv"
    
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

    with mlflow.start_run(run_name="CI_Training_Run") as run:
        print(f"Memulai run: {run.info.run_id}")
        
        print("Melatih model...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Accuracy: {accuracy:.4f}")

        print("Logging parameter dan metrik secara manual...")
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        print("Menyimpan dan logging artefak model secara manual...")
        model_filename = "model.pkl"
        joblib.dump(model, model_filename)

        mlflow.log_artifact(model_filename, artifact_path="model") 

        os.remove(model_filename)
        
        print("Pelatihan CI selesai. Artefak di-log secara manual.")

        run_id = run.info.run_id
        with open("../run_id.txt", "w") as f:
            f.write(run_id)
        print(f"Menyimpan Run ID: {run_id}")

if __name__ == "__main__":
    main()