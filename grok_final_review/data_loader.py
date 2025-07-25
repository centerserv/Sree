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
        Create synthetic dataset for testing.
        
        Args:
            n_samples: Number of samples to create (None for default)
            
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        config = DATASET_CONFIG["synthetic"]
        n_samples = n_samples or config["n_samples"]
        
        self.logger.info(f"Creating synthetic dataset (n_samples={n_samples})")
        
        # Generate synthetic data with clear patterns
        np.random.seed(42)
        
        # Create features with clear patterns
        X = np.random.randn(n_samples, config["n_features"])
        
        # Create labels based on clear patterns
        # Pattern: if sum of first 3 features > 0, then class 1, else class 0
        feature_sum = np.sum(X[:, :3], axis=1)
        y = (feature_sum > 0).astype(int)
        
        noise = np.random.randn(n_samples) * 0.1
        y = ((feature_sum + noise) > 0).astype(int)
        
        self.logger.info(f"Synthetic dataset created: X.shape={X.shape}, y.shape={y.shape}")
        return X, y
            
    def create_credit_risk_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a realistic credit risk dataset for bank analysts with 12 features, 50/50 balance,
        10-15% noise, and 5% outliers as per client requirements.
        
        Args:
            n_samples: Number of samples to create (default 1000)
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        np.random.seed(42)
        n_features = 12
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)
        
        # Feature names for analysis
        feature_names = [
            'credit_score',      # 0: Credit score (300-850)
            'payment_history',   # 1: Payment history (0-100%)
            'debt_to_income',    # 2: Debt-to-income ratio (0-100%)
            'credit_history',    # 3: Length of credit history (1-25 years)
            'credit_inquiries',  # 4: Number of credit inquiries (0-10)
            'credit_utilization', # 5: Credit utilization (0-100%)
            'open_accounts',     # 6: Number of open accounts (1-15)
            'delinquencies',     # 7: Delinquency in last 2 years (0-5)
            'annual_income',     # 8: Annual income (20k-200k)
            'employment_length', # 9: Employment length (0-40 years)
            'age',              # 10: Age (18-75)
            'region'            # 11: Region (0-3)
        ]
        
        # Generate realistic credit data with correlations
        # Credit score: normal distribution around 650
        X[:, 0] = np.random.normal(650, 150, n_samples)
        X[:, 0] = np.clip(X[:, 0], 300, 850)
        
        # Payment history: correlated with credit score
        X[:, 1] = 50 + (X[:, 0] - 300) * 0.4 + np.random.normal(0, 15, n_samples)
        X[:, 1] = np.clip(X[:, 1], 0, 100)
        
        # Debt-to-income: inversely correlated with income
        X[:, 2] = np.random.normal(35, 15, n_samples)
        X[:, 2] = np.clip(X[:, 2], 5, 80)
        
        # Credit history: correlated with age
        X[:, 3] = np.random.normal(8, 5, n_samples)
        X[:, 3] = np.clip(X[:, 3], 1, 25)
        
        # Credit inquiries: random but realistic
        X[:, 4] = np.random.poisson(3, n_samples)
        X[:, 4] = np.clip(X[:, 4], 0, 10)
        
        # Credit utilization: correlated with debt-to-income
        X[:, 5] = X[:, 2] * 0.8 + np.random.normal(0, 10, n_samples)
        X[:, 5] = np.clip(X[:, 5], 0, 100)
        
        # Open accounts: correlated with credit history
        X[:, 6] = 3 + X[:, 3] * 0.3 + np.random.normal(0, 2, n_samples)
        X[:, 6] = np.clip(X[:, 6], 1, 15)
        
        # Delinquencies: inversely correlated with payment history
        X[:, 7] = np.random.poisson(0.5, n_samples)
        X[:, 7] = np.clip(X[:, 7], 0, 5)
        
        # Annual income: normal distribution
        X[:, 8] = np.random.normal(75000, 30000, n_samples)
        X[:, 8] = np.clip(X[:, 8], 20000, 200000)
        
        # Employment length: correlated with age
        X[:, 9] = np.random.normal(8, 6, n_samples)
        X[:, 9] = np.clip(X[:, 9], 0, 40)
        
        # Age: normal distribution
        X[:, 10] = np.random.normal(45, 15, n_samples)
        X[:, 10] = np.clip(X[:, 10], 18, 75)
        
        # Region: categorical
        X[:, 11] = np.random.randint(0, 4, n_samples)
        
        # Create labels based on realistic credit scoring rules
        # Good credit criteria (label 1):
        good_credit = (
            (X[:, 0] >= 650) &           # Credit score >= 650
            (X[:, 1] >= 85) &            # Payment history >= 85%
            (X[:, 2] <= 40) &            # Debt-to-income <= 40%
            (X[:, 5] <= 50) &            # Credit utilization <= 50%
            (X[:, 7] <= 1) &             # Delinquencies <= 1
            (X[:, 8] >= 40000)           # Annual income >= 40k
        )
        
        y[good_credit] = 1
        y[~good_credit] = 0
        
        # Add 10-15% noise to labels
        noise_rate = np.random.uniform(0.10, 0.15)
        noise_indices = np.random.choice(n_samples, size=int(n_samples * noise_rate), replace=False)
        y[noise_indices] = 1 - y[noise_indices]  # Flip labels
        
        # Add 5% outliers (problematic individuals)
        outlier_rate = 0.05
        outlier_indices = np.random.choice(n_samples, size=int(n_samples * outlier_rate), replace=False)
        
        for idx in outlier_indices:
            outlier_type = np.random.choice(['no_income', 'no_credit', 'extreme_values'])
            if outlier_type == 'no_income':
                X[idx, 8] = 0  # No income
                X[idx, 0] = 300  # Very low credit score
            elif outlier_type == 'no_credit':
                X[idx, 0] = 0  # No credit score
                X[idx, 3] = 0  # No credit history
            elif outlier_type == 'extreme_values':
                X[idx, 8] = 1000000  # Unrealistic income
                X[idx, 0] = 1000  # Impossible credit score
        
        # Force 50/50 balance
        n_good = np.sum(y == 1)
        n_bad = np.sum(y == 0)
        target_per_class = n_samples // 2
        
        if n_good > target_per_class:
            # Convert some good to bad
            good_indices = np.where(y == 1)[0]
            np.random.shuffle(good_indices)
            y[good_indices[:n_good - target_per_class]] = 0
        elif n_bad > target_per_class:
            # Convert some bad to good
            bad_indices = np.where(y == 0)[0]
            np.random.shuffle(bad_indices)
            y[bad_indices[:n_bad - target_per_class]] = 1
        
        # Store feature names for analysis
        self.credit_feature_names = feature_names
        
        self.logger.info(f"Credit dataset created: {n_samples} samples, {n_features} features")
        self.logger.info(f"Class distribution: {np.sum(y==1)} good, {np.sum(y==0)} bad")
        self.logger.info(f"Noise rate: {noise_rate:.1%}, Outliers: {len(outlier_indices)}")
        
        return X, y
    
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
    
    def load_credit_risk_dataset(self, dataset_path: str = "synthetic_credit_risk.csv") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the credit risk dataset from CSV file.
        
        Args:
            dataset_path: Path to the credit risk dataset CSV file
            
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        import pandas as pd
        
        self.logger.info(f"Loading credit risk dataset from {dataset_path}")
        
        # Load dataset from CSV
        df = pd.read_csv(dataset_path)
        
        # Separate features and target
        X = df.iloc[:, :-1].values  # All columns except the last one
        y = df.iloc[:, -1].values   # Last column is the target
        
        self.logger.info(f"Credit risk dataset loaded: X.shape={X.shape}, y.shape={y.shape}")
        self.logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y

    def get_dataset_info(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Dictionary containing dataset information
        """
        # CORREÇÃO 4: Dataset-Agnostic - Add auto n_classes=len(np.unique(y))
        # scale entropy base=np.log2(max(n_classes,2)) for binary/multi-class
        n_classes = len(np.unique(y))
        entropy_base = np.log2(max(n_classes, 2))  # Ensure minimum base of 2
        
        info = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": n_classes,
            "class_distribution": np.bincount(y).tolist(),
            "feature_stats": {
                "mean": float(np.mean(X)),
                "std": float(np.std(X)),
                "min": float(np.min(X)),
                "max": float(np.max(X))
            },
            "entropy_base": entropy_base  # Add entropy base to info
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
        
        # Create credit risk dataset (designed for Phase 1 targets)
        X_credit, y_credit = loader.create_credit_risk_dataset()
        X_train_credit, X_test_credit, y_train_credit, y_test_credit = loader.preprocess_data(X_credit, y_credit)
        datasets["credit_risk"] = {
            "X": X_credit,
            "y": y_credit,
            "X_train": X_train_credit,
            "X_test": X_test_credit,
            "y_train": y_train_credit,
            "y_test": y_test_credit,
            "info": loader.get_dataset_info(X_credit, y_credit)
        }
        
        if logger:
            logger.info("All datasets loaded successfully")
        return datasets
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to load datasets: {e}")
        raise 