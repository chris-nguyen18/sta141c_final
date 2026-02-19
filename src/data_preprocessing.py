import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

def load_credit_card_dataset():
   base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   file_path = os.path.join(base_dir, "data", "raw", "default_of_credit_card_clients.csv")
   
   df = pd.read_csv(file_path, header=1)
   df.columns = df.columns.str.strip()

   if "ID" in df.columns:
      df = df.drop(columns=["ID"])

   # Separate features and target
   X = df.drop(columns=["default payment next month"])
   y = df["default payment next month"]

   return X, y

def clean_data(X, y):
   X = X.fillna(0) 
   y = y.astype(int)  

   return X, y

def split_data(X, y, test_size=0.2, random_state=67):
   X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state
   )
      
   return X_train, X_test, y_train, y_test
   

def save_processed_data(X_train, X_test, y_train, y_test, folder="data/processed"):
   os.makedirs(folder, exist_ok=True)
   np.save(os.path.join(folder, "X_train.npy"), X_train)
   np.save(os.path.join(folder, "X_test.npy"), X_test)
   np.save(os.path.join(folder, "y_train.npy"), y_train)
   np.save(os.path.join(folder, "y_test.npy"), y_test)

if __name__ == "__main__":
   X, y = load_credit_card_dataset()
   X, y = clean_data(X, y)

   X_train, X_test, y_train, y_test = split_data(X, y)
   save_processed_data(X_train, X_test, y_train, y_test)

   # metadata 
   #print(dataset.metadata) 
   # variable information 
   #print(dataset.variables) 

   

