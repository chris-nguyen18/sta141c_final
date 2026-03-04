import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from feature_engineering import load_processed_data

def save_model(model, filename):
   base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   model_dir = os.path.join(base_dir, "models")
   os.makedirs(model_dir, exist_ok=True)
   joblib.dump(model, os.path.join(model_dir, filename))

if __name__ == "__main__":
   X_train, X_test, y_train, y_test = load_processed_data()

   # Logistic Regression Only
   pipe_logistic_raw = Pipeline([
      ('scaler', StandardScaler()),
      ('classifier', LogisticRegression(max_iter=1000, class_weight="balanced"))
   ])
   pipe_logistic_raw.fit(X_train, y_train)
   save_model(pipe_logistic_raw, "logistic_raw.joblib")

   # Logistic Regression w/ PCA
   # Scaler + PCA + Model
   pipe_logistic_pca = Pipeline([
      ('scaler', StandardScaler()),
      ('pca', PCA(n_components=15)),
      ('classifier', LogisticRegression(max_iter=1000, class_weight="balanced"))
   ])
   pipe_logistic_pca.fit(X_train, y_train)
   save_model(pipe_logistic_pca, "logistic_pca.joblib")

   # Random Forest
   pipe_rf = Pipeline([
      ('scaler', StandardScaler()),
      ('classifier', RandomForestClassifier(n_estimators=100, random_state=67, n_jobs=-1))
   ])
   pipe_rf.fit(X_train, y_train)
   save_model(pipe_rf, "random_forest.joblib")

   print("Training complete with Pipelines.")