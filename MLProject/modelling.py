import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path):
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
    mlflow.sklearn.autolog()

    # Tentukan model terbaik 
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    with mlflow.start_run(run_name="CI_Training_Run") as run:
        print(f"Memulai run: {run.info.run_id}")
        
        print("Melatih model...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print("Pelatihan CI selesai. Model di-log secara otomatis.")

if __name__ == "__main__":
    main()
