from feature_engineering import (
   load_processed_data,
   scale_features,
   apply_pca
)

from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_logistic(X_train, y_train):
   model = LogisticRegression(max_iter=1000)
   model.fit(X_train, y_train)
   return model


if __name__ == "__main__":
   # Load data and scale
   X_train, X_test, y_train, y_test = load_processed_data()
   X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

   # Train logistic (no PCA)
   model_raw = train_logistic(X_train_scaled, y_train)
   # Apply PCA
   X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
   # Train logistic (PCA)
   model_pca = train_logistic(X_train_pca, y_train)

   print("Training complete.")
