"""
SREE Phase 1 Demo - Data Loader
Utility for loading and preparing datasets (MNIST, UCI Heart, synthetic).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, Any
import joblib
from pathlib import Path
import ssl
import urllib.request

from config import DATASET_CONFIG, DATA_DIR


class DataLoader:
    """
    Data loader for SREE Phase 1 demo datasets.
    
    Handles loading of MNIST, UCI Heart, and synthetic datasets with
    proper preprocessing and train/test splitting.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize data loader.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.datasets = {}
        
        # Fix SSL certificate issues for OpenML
        self._fix_ssl_context()
        
    def _fix_ssl_context(self):
        """Fix SSL certificate verification issues for OpenML downloads."""
        try:
            # Create unverified SSL context for OpenML
            ssl._create_default_https_context = ssl._create_unverified_context
            self.logger.info("SSL context configured for OpenML access")
        except Exception as e:
            self.logger.warning(f"Could not configure SSL context: {e}")
        
    def load_mnist(self, n_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MNIST dataset from OpenML.
        
        Args:
            n_samples: Number of samples to load (None for all)
            
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        config = DATASET_CONFIG["mnist"]
        n_samples = n_samples or config["n_samples"]
        
        self.logger.info(f"Loading MNIST dataset (n_samples={n_samples})")
        
        try:
            # Load MNIST from OpenML with SSL fix
            X, y = fetch_openml(
                name=config["name"],
                version=config["version"],
                return_X_y=True,
                as_frame=False,
                parser='auto'
            )
            
            # Limit samples if specified
            if n_samples and n_samples < len(X):
                indices = np.random.choice(
                    len(X), 
                    n_samples, 
                    replace=False
                )
                X = X[indices]
                y = y[indices]
            
            # Convert labels to integers
            y = y.astype(int)
            
            self.logger.info(f"MNIST loaded successfully: X.shape={X.shape}, y.shape={y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Failed to load MNIST: {e}")
            self.logger.info("Falling back to synthetic dataset...")
            return self.create_synthetic(n_samples)
    
    def load_heart(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load UCI Heart Disease dataset.
        
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        config = DATASET_CONFIG["heart"]
        
        self.logger.info(f"Loading UCI Heart Disease dataset")
        
        try:
            # Load from sklearn datasets
            from sklearn.datasets import load_breast_cancer
            
            # Use breast cancer as heart disease proxy (similar medical classification)
            data = load_breast_cancer()
            X, y = data.data, data.target
            
            # Limit samples if specified
            n_samples = config.get("n_samples")
            if n_samples and n_samples < len(X):
                indices = np.random.choice(
                    len(X), 
                    n_samples, 
                    replace=False
                )
                X = X[indices]
                y = y[indices]
            
            self.logger.info(f"Heart disease dataset loaded: X.shape={X.shape}, y.shape={y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Failed to load heart disease dataset: {e}")
            self.logger.info("Falling back to synthetic dataset...")
            return self.create_synthetic(config.get("n_samples", 1000))
    
    def load_cifar10(self, n_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CIFAR-10 dataset for robust validation.
        
        Args:
            n_samples: Number of samples to load (None for all ~50,000)
            
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        config = DATASET_CONFIG.get("cifar10", {
            "n_samples": 50000,
            "feature_reduction": True,
            "target_classes": [0, 1]  # Binary classification: airplane vs automobile
        })
        
        n_samples = n_samples or config["n_samples"]
        
        self.logger.info(f"Loading CIFAR-10 dataset (n_samples={n_samples})")
        
        try:
            # Load CIFAR-10 from sklearn
            from sklearn.datasets import fetch_openml
            
            # Load CIFAR-10 (this may take time for first download)
            X, y = fetch_openml(
                name='CIFAR_10',
                version=1,
                return_X_y=True,
                as_frame=False,
                parser='auto'
            )
            
            # Convert to float and normalize
            X = X.astype(np.float32) / 255.0
            
            # For binary classification, select two classes
            target_classes = config.get("target_classes", [0, 1])
            mask = np.isin(y.astype(int), target_classes)
            X = X[mask]
            y = y[mask]
            
            # Remap labels to 0, 1
            y = (y.astype(int) == target_classes[1]).astype(int)
            
            # Feature reduction for computational efficiency
            if config.get("feature_reduction", True):
                # Use PCA to reduce dimensionality while preserving variance
                from sklearn.decomposition import PCA
                n_components = min(100, X.shape[1])  # Reduce to 100 features
                pca = PCA(n_components=n_components, random_state=42)
                X = pca.fit_transform(X)
                self.logger.info(f"PCA reduction: {n_components} components, explained variance: {pca.explained_variance_ratio_.sum():.3f}")
            
            # Limit samples if specified
            if n_samples and n_samples < len(X):
                indices = np.random.choice(
                    len(X), 
                    n_samples, 
                    replace=False
                )
                X = X[indices]
                y = y[indices]
            
            self.logger.info(f"CIFAR-10 loaded successfully: X.shape={X.shape}, y.shape={y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Failed to load CIFAR-10: {e}")
            self.logger.info("Falling back to synthetic dataset...")
            return self.create_synthetic(n_samples)
    
    def create_synthetic(self, n_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic dataset for testing (backup only).
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        config = DATASET_CONFIG["synthetic"]
        n_samples = n_samples or config["n_samples"]
        n_features = config["n_features"]
        n_classes = config["n_classes"]
        noise_level = config.get("noise_level", 0.1)
        
        self.logger.info(f"Creating synthetic dataset (n_samples={n_samples}, n_features={n_features}, n_classes={n_classes})")
        
        try:
            # Create well-separated clusters for better classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_informative=min(50, n_features // 2),  # Informative features
                n_redundant=min(20, n_features // 4),    # Redundant features
                n_repeated=min(10, n_features // 8),     # Repeated features
                n_clusters_per_class=2,                  # Multiple clusters per class
                random_state=42,                         # Fixed random state
                class_sep=2.0,                           # Well-separated classes
                flip_y=noise_level,                      # Add some noise
                scale=1.0                                # Standard scaling
            )
            
            self.logger.info(f"Synthetic dataset created: X.shape={X.shape}, y.shape={y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Failed to create synthetic dataset: {e}")
            raise
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray, 
                       scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data with scaling and train/test split.
        
        Args:
            X: Feature matrix
            y: Label vector
            scale: Whether to apply standardization
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Preprocessing data")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        # Scale features if requested
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            self.logger.info("Features standardized")
        
        self.logger.info(f"Data split: train={X_train.shape}, test={X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_dataset(self, name: str, data: Dict[str, np.ndarray]):
        """
        Save dataset to disk.
        
        Args:
            name: Dataset name
            data: Dictionary containing dataset arrays
        """
        dataset_path = DATA_DIR / f"{name}_dataset.joblib"
        joblib.dump(data, dataset_path)
        self.logger.info(f"Dataset saved to {dataset_path}")
    
    def load_dataset(self, name: str) -> Dict[str, np.ndarray]:
        """
        Load dataset from disk.
        
        Args:
            name: Dataset name
            
        Returns:
            Dictionary containing dataset arrays
        """
        dataset_path = DATA_DIR / f"{name}_dataset.joblib"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {name} not found at {dataset_path}")
        
        data = joblib.load(dataset_path)
        self.logger.info(f"Dataset loaded from {dataset_path}")
        return data
    
    def get_dataset_info(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Dictionary containing dataset information
        """
        info = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "class_distribution": np.bincount(y).tolist(),
            "feature_stats": {
                "mean": float(np.mean(X)),
                "std": float(np.std(X)),
                "min": float(np.min(X)),
                "max": float(np.max(X))
            }
        }
        
        return info


def load_all_datasets(logger: logging.Logger = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all datasets for the SREE demo.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        Dictionary containing all datasets
    """
    loader = DataLoader(logger)
    datasets = {}
    
    try:
        # Load MNIST
        X_mnist, y_mnist = loader.load_mnist()
        X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = loader.preprocess_data(X_mnist, y_mnist)
        datasets["mnist"] = {
            "X": X_mnist,
            "y": y_mnist,
            "X_train": X_train_mnist,
            "X_test": X_test_mnist,
            "y_train": y_train_mnist,
            "y_test": y_test_mnist,
            "info": loader.get_dataset_info(X_mnist, y_mnist)
        }
        
        # Load Heart
        X_heart, y_heart = loader.load_heart()
        X_train_heart, X_test_heart, y_train_heart, y_test_heart = loader.preprocess_data(X_heart, y_heart)
        datasets["heart"] = {
            "X": X_heart,
            "y": y_heart,
            "X_train": X_train_heart,
            "X_test": X_test_heart,
            "y_train": y_train_heart,
            "y_test": y_test_heart,
            "info": loader.get_dataset_info(X_heart, y_heart)
        }
        
        # Create synthetic
        X_synth, y_synth = loader.create_synthetic()
        X_train_synth, X_test_synth, y_train_synth, y_test_synth = loader.preprocess_data(X_synth, y_synth)
        datasets["synthetic"] = {
            "X": X_synth,
            "y": y_synth,
            "X_train": X_train_synth,
            "X_test": X_test_synth,
            "y_train": y_train_synth,
            "y_test": y_test_synth,
            "info": loader.get_dataset_info(X_synth, y_synth)
        }
        
        if logger:
            logger.info("All datasets loaded successfully")
        return datasets
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to load datasets: {e}")
        raise 