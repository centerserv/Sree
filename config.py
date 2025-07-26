"""
SREE Phase 1 Demo - Configuration
Central configuration file for logging, datasets, and model parameters.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
PLOTS_DIR = PROJECT_ROOT / "plots"
TESTS_DIR = PROJECT_ROOT / "tests"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, TESTS_DIR]:
    directory.mkdir(exist_ok=True)

# Logging configuration
def setup_logging(level: str = "INFO", log_file: str = "sree_demo.log") -> logging.Logger:
    """
    Set up logging configuration for the SREE demo.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Name of log file in logs directory
        
    Returns:
        Configured logger instance
    """
    log_path = LOGS_DIR / log_file
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logger
    logger = logging.getLogger('SREE_Demo')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Create console handler only for non-test environments
    if not os.environ.get('PYTEST_CURRENT_TEST'):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.addHandler(file_handler)
    
    return logger

# Dataset configuration
DATASET_CONFIG = {
    "mnist": {
        "name": "mnist_784",
        "version": 1,
        "n_samples": 1000,
        "target_digits": [0, 1]  # Binary classification: 0 vs 1
    },
    "heart": {
        "n_samples": 569,  # Full UCI Heart dataset
        "target_classes": [0, 1]  # Binary classification
    },
    "synthetic": {
        "n_samples": 2000,
        "n_features": 100,
        "n_classes": 5,
        "noise_level": 0.1,
        "class_sep": 2.0
    },
    "cifar10": {
        "n_samples": 50000,  # Full CIFAR-10 for robust validation
        "feature_reduction": True,
        "n_components": 100,  # PCA components
        "target_classes": [0, 1],  # Binary: airplane vs automobile
        "normalize": True
    }
}

# Model configuration
MODEL_CONFIG = {
    "mlp": {
        "hidden_layer_sizes": (128, 64),  # Simplified architecture for better convergence
        "max_iter": 1500,  # More iterations for convergence
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "learning_rate_init": 0.005,  # Higher learning rate for faster convergence
        "alpha": 0.0001,  # Slightly more regularization
        "tol": 1e-4,  # Higher tolerance for convergence
        "activation": "relu",
        "solver": "adam"
    },
    "xgboost": {
        "n_estimators": 300,  # More trees for better performance
        "max_depth": 8,  # Slightly deeper trees
        "learning_rate": 0.05,  # Lower learning rate for better generalization
        "random_state": 42,
        "subsample": 0.9,  # Higher subsample for better performance
        "colsample_bytree": 0.9,  # Higher colsample for better performance
        "reg_alpha": 0.05,  # L1 regularization
        "reg_lambda": 0.1,  # L2 regularization
        "min_child_weight": 3,  # Prevent overfitting
        "gamma": 0.1,  # Minimum loss reduction for split
        "scale_pos_weight": 1.0  # Handle class imbalance
    },
    "default_model": "xgboost"  # Use XGBoost as default
}

# PPP loop configuration
PPP_CONFIG = {
    "iterations": 30,  # Increased for 10-20 iterations convergence target
    "gamma": 0.4,      # State update rate (increased for faster convergence)
    "alpha": 0.4,      # Trust update rate (increased for faster convergence)
    "beta": 0.7,       # Permanence weight (increased for more blocks)
    "delta": 0.4,      # Logic weight (increased for better validation)
    "initial_trust": 0.85, # Higher initial trust for better starting point
    "initial_state": 0.80, # Higher initial state
    "presence": {
        "entropy_threshold": 1.5,    # Lower entropy threshold for more refinement
        "min_confidence": 0.5,  # Aumentado para forçar mais confiança
        "entropy_penalty": 5.0,  # Penalização mais agressiva
        "refinement_factor": 0.85    # More aggressive refinement
    },
    "permanence": {
        "hash_algorithm": "sha256",  # Hash algorithm for logging
        "block_size": 40,            # Smaller block size for more blocks
        "consistency_threshold": 0.75 # Lower threshold for more blocks
    },
    "logic": {
        "consistency_weight": 0.7,   # Higher weight for consistency validation
        "confidence_threshold": 0.60, # Lower threshold for more processing
        "max_inconsistencies": 0.20  # Fewer allowed inconsistencies
    }
}

# Testing configuration
TEST_CONFIG = {
    "fault_injection": {
        "corruption_rate": 0.15,  # 15% label corruption
        "random_state": 42
    },
    "ablation": {
        "test_combinations": [
            ["pattern"],
            ["pattern", "presence"],
            ["pattern", "permanence"],
            ["pattern", "logic"],
            ["pattern", "presence", "permanence"],
            ["pattern", "presence", "permanence", "logic"]  # Full PPP
        ]
    },
    "cross_validation": {
        "n_splits": 10,
        "random_state": 42
    }
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "figure_size": (8, 6),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "save_format": "png"
}

# Target metrics (from manuscript Table 3)
TARGET_METRICS = {
    "phase1": {
        "accuracy": 0.85,   # ~85% (Phase 1 target with simulated quantum/blockchain)
        "trust": 0.85       # T ≈ 0.85 (Phase 1 target)
    },
    "phase2": {
        "accuracy": 0.985,  # 98.5% (Phase 2 target with real Qiskit/Ganache)
        "trust": 0.96       # T ≈ 0.96 (Phase 2 target)
    },
    "baselines": {
        "ai_only": {"accuracy": 0.85, "trust": 0.72},
        "rlhf": {"accuracy": 0.901, "trust": 0.79},
        "chainlink": {"accuracy": 0.887, "trust": 0.81},
        "qaoa": {"accuracy": 0.893, "trust": 0.82}
    }
}

# Phase 1 implementation details
PHASE1_CONFIG = {
    "implementation": {
        "quantum": "NumPy simulation (educational foundation)",
        "blockchain": "hashlib simulation (concept validation)",
        "purpose": "Educational demonstration and academic validation"
    },
    "performance": {
        "accuracy_target": 0.85,  # ~85%
        "trust_target": 0.85,     # T ≈ 0.85
        "datasets": ["MNIST", "UCI Heart Disease"]
    }
}

def get_config() -> Dict[str, Any]:
    """
    Get complete configuration dictionary.
    
    Returns:
        Dictionary containing all configuration settings
    """
    return {
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "logs_dir": str(LOGS_DIR),
            "plots_dir": str(PLOTS_DIR),
            "tests_dir": str(TESTS_DIR)
        },
        "datasets": DATASET_CONFIG,
        "model": MODEL_CONFIG,
        "ppp": PPP_CONFIG,
        "testing": TEST_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "targets": TARGET_METRICS,
        "phase1": PHASE1_CONFIG
    } 