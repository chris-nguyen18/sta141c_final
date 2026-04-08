import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_processed_data():
   base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   folder = os.path.join(base_dir, "data", "processed")

   X_train = pd.read_pickle(os.path.join(folder, "X_train.pkl"))
   X_test = pd.read_pickle(os.path.join(folder, "X_test.pkl"))
   y_train = pd.read_pickle(os.path.join(folder, "y_train.pkl"))
   y_test = pd.read_pickle(os.path.join(folder, "y_test.pkl"))
   
   return X_train, X_test, y_train, y_test


def get_preprocessor(X):
   # Identify columns by type
   categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
   # Everything else is numerical 
   numerical_cols = [col for col in X.columns if col not in categorical_cols]

   num_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler())
   ])

   cat_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
   ])

   preprocessor = ColumnTransformer(
      transformers=[
         #('num', StandardScaler(), numerical_cols),
         #('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
         ('num', num_transformer, numerical_cols),
         ('cat', cat_transformer, categorical_cols)
      ]
   )
   
   return preprocessor


def save_model(model, filename):
   base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   model_dir = os.path.join(base_dir, "models")
   os.makedirs(model_dir, exist_ok=True)
   joblib.dump(model, os.path.join(model_dir, filename))


if __name__ == "__main__":
   X_train, X_test, y_train, y_test = load_processed_data()
   preprocessor = get_preprocessor(X_train)
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=67)

   # Logistic Regression Only
   pipe_logistic_raw = Pipeline([
      ('preprocessor', preprocessor),
      ('classifier', LogisticRegression(max_iter=10000, class_weight="balanced", penalty='l2'))
   ])

   print("Evaluating logistic model with Cross-Validation...")
   cv_results = cross_val_score(pipe_logistic_raw, X_train, y_train, cv=skf, scoring='recall')
   print(f"Logistic Mean Recall Score: {cv_results.mean():.4f}")

   pipe_logistic_raw.fit(X_train, y_train)
   save_model(pipe_logistic_raw, "logistic_raw.joblib")

   # Logistic Regression w/ PCA
   # Scaler + PCA + Model
   pipe_logistic_pca = Pipeline([
      ('preprocessor', preprocessor),
      ('pca', PCA(n_components=15)),
      ('classifier', LogisticRegression(max_iter=10000, class_weight="balanced",penalty='l2'))
   ])

   print("Evaluating logistic-pca model with Cross-Validation...")
   cv_results = cross_val_score(pipe_logistic_pca, X_train, y_train, cv=skf, scoring='recall')
   print(f"Logistic-PCA Mean Recall Score: {cv_results.mean():.4f}")

   pipe_logistic_pca.fit(X_train, y_train)
   save_model(pipe_logistic_pca, "logistic_pca.joblib")

   # Random Forest
   pipe_rf = Pipeline([
      ('preprocessor', preprocessor),
      ('classifier', RandomForestClassifier(n_estimators=100, class_weight="balanced", 
                                            random_state=67, n_jobs=-1))
   ])

   print("Evaluating random forest model with Cross-Validation...")
   cv_results = cross_val_score(pipe_logistic_raw, X_train, y_train, cv=skf, scoring='recall')
   print(f"Random Forest Mean Recall Score: {cv_results.mean():.4f}")

   pipe_rf.fit(X_train, y_train)
   save_model(pipe_rf, "random_forest.joblib")

   print("Training complete with Pipelines.")