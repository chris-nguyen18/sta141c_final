from feature_engineering import (
   load_processed_data,
   scale_features,
   apply_pca
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_logistic(X_train, y_train):
   model = LogisticRegression(max_iter=1000)
   model.fit(X_train, y_train)
   return model

def train_random_forest(X_train, y_train):
   model = RandomForestClassifier(
      n_estimators=100,
      random_state=67,
      n_jobs=-1
   )
   model.fit(X_train, y_train)
   return model

def save_model(model, filename):
   base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   model_dir = os.path.join(base_dir, "models")

   os.makedirs(model_dir, exist_ok=True)
   joblib.dump(model, os.path.join(model_dir, filename))

if __name__ == "__main__":
   # Load data and scale
   X_train, X_test, y_train, y_test = load_processed_data()
   X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

   save_model(scaler, "scaler.joblib")

   # Train logistic (no PCA)
   log_model_raw = train_logistic(X_train_scaled, y_train)
   save_model(scaler, "scaler.joblib")
   
   # Apply PCA
   X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
   save_model(pca, "pca.joblib")
   
   # Train logistic (PCA)
   log_model_pca = train_logistic(X_train_pca, y_train)
   save_model(log_model_pca, "logistic_pca.joblib")

   # Random forest
   model_rf = train_random_forest(X_train_scaled, y_train)
   save_model(model_rf, "random_forest.joblib")

   print("Training complete.")
