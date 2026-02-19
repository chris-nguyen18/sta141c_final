import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_processed_data():
   base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   folder = os.path.join(base_dir, "data", "processed")

   X_train = np.load(os.path.join(folder, "X_train.npy"))
   X_test = np.load(os.path.join(folder, "X_test.npy"))
   y_train = np.load(os.path.join(folder, "y_train.npy"))
   y_test = np.load(os.path.join(folder, "y_test.npy"))
   
   return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   return X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train_scaled, X_test_scaled, n_components=0.95):
   pca = PCA(n_components=n_components)
   X_train_pca = pca.fit_transform(X_train_scaled)
   X_test_pca = pca.transform(X_test_scaled)
   return X_train_pca, X_test_pca, pca
