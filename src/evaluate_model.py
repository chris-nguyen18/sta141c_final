import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
   accuracy_score, precision_score, 
   recall_score, f1_score, roc_auc_score, 
   confusion_matrix, ConfusionMatrixDisplay
   )

def evaluate_model(model, X_test, y_test, output_name):
   #model = joblib.load(model_path)

   y_pred = model.predict(X_test)
   y_proba = model.predict_proba(X_test)[:, 1]

   metrics = {
      "Accuracy": accuracy_score(y_test, y_pred),
      "Precision": precision_score(y_test, y_pred),
      "Recall": recall_score(y_test, y_pred),
      "F1": f1_score(y_test, y_pred),
      "ROC-AUC": roc_auc_score(y_test, y_proba)
   }

   
   cm = confusion_matrix(y_test, y_pred)
   disp = ConfusionMatrixDisplay(confusion_matrix=cm)
   disp.plot()
   plt.title(f"Confusion Matrix: {output_name}")
   plt.savefig(f"../reports/{output_name}_cm.png")
   plt.close()

   return metrics

if __name__ == "__main__":
   # Load test data
   X_test = np.load("../data/processed/X_test.npy")
   y_test = np.load("../data/processed/y_test.npy")

   # Load trained model
   models = {
      "logistic_raw": "../models/logistic_raw.joblib",
      "logistic_pca": "../models/logistic_pca.joblib",
      "random_forest": "../models/random_forest.joblib"
   } 

   
   for name, path in models.items():
      print(f"\nEvaluating {name}...")
      model = joblib.load(path)

      if hasattr(model, "predict"):
         metrics = evaluate_model(model, X_test, y_test, name)

         for k, v in metrics.items():
            print(f"{k}: {v}")
      else:
        # This will tell you exactly why it's skipping
        print(f"SKIPPED: {name} is actually a {type(model).__name__}")
 