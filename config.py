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
        "entropy_threshold": 2.2,    # Higher threshold - flag only very high entropy
        "min_confidence": 0.3,  # Lower threshold - flag only very low confidence
        "entropy_penalty": 3.0,  # Moderate penalty
        "refinement_factor": 0.90    # Less aggressive refinement
    },
    "permanence": {
        "hash_algorithm": "sha256",  # Hash algorithm for logging
        "block_size": 40,            # Smaller block size for more blocks
        "min_blocks": 2,             # Minimum blocks required
        "max_deviation": 0.02,       # Maximum allowed deviation
        "convergence_threshold": 0.98, # Higher convergence threshold
        "consistency_threshold": 0.75  # Consistency threshold for permanence validation
    },
    "logic": {
        "min_conditions": 3,         # Minimum logical conditions
        "max_rules": 15,             # Maximum logical rules
        "confidence_threshold": 0.75, # Lower threshold - flag only very low confidence
        "support_threshold": 0.25,   # Higher support threshold - fewer rules
        "consistency_weight": 0.7,   # Higher weight for consistency validation
        "max_inconsistencies": 0.30  # More allowed inconsistencies
    }
}

# Dashboard optimized PPP configuration for faster processing
DASHBOARD_PPP_CONFIG = {
    "iterations": 8,   # Reduced from 30 to 8 for faster dashboard response
    "gamma": 0.5,      # Slightly higher for faster convergence
    "alpha": 0.5,      # Slightly higher for faster convergence  
    "beta": 0.6,       # Reduced slightly for speed
    "delta": 0.3,      # Reduced for speed
    "initial_trust": 0.85, # Same as original
    "initial_state": 0.80, # Same as original
    "presence": {
        "entropy_threshold": 2.0,    # Higher threshold - flag only very high entropy
        "min_confidence": 0.35,      # Lower threshold - flag only very low confidence
        "entropy_penalty": 2.5,      # Moderate penalty
        "refinement_factor": 0.85    # Less aggressive refinement
    },
    "permanence": {
        "hash_algorithm": "sha256",
        "block_size": 50,            # Larger block size for speed
        "min_blocks": 2,
        "max_deviation": 0.03,       # Slightly more tolerant
        "convergence_threshold": 0.95, # Slightly lower for speed
        "consistency_threshold": 0.70  # Lower threshold for faster processing
    },
    "logic": {
        "min_conditions": 2,         # Reduced for speed
        "max_rules": 10,             # Reduced for speed
        "confidence_threshold": 0.70, # Lower threshold - flag only very low confidence
        "support_threshold": 0.30,   # Higher support threshold - fewer rules
        "consistency_weight": 0.6,   # Lower weight for speed
        "max_inconsistencies": 0.35  # More tolerant for speed
    }
}

# Ultra-fast configuration for small datasets (100 rows or less)
ULTRA_FAST_CONFIG = {
    "iterations": 4,   # Balanced iterations for speed and quality
    "gamma": 0.6,      # Balanced for convergence and quality
    "alpha": 0.6,      # Balanced for convergence and quality
    "beta": 0.6,       # Balanced permanence processing
    "delta": 0.3,      # Balanced logic processing
    "initial_trust": 0.85,
    "initial_state": 0.80,
    "presence": {
        "entropy_threshold": 2.2,    # Maintain quality threshold
        "min_confidence": 0.30,      # Maintain quality threshold
        "entropy_penalty": 2.5,      # Balanced penalty
        "refinement_factor": 0.85    # Balanced refinement
    },
    "permanence": {
        "hash_algorithm": "md5",     # Faster hash algorithm
        "block_size": 100,           # Large block size for speed
        "min_blocks": 1,             # Minimal blocks
        "max_deviation": 0.03,       # Maintain quality
        "convergence_threshold": 0.95, # Maintain quality
        "consistency_threshold": 0.75  # Maintain quality
    },
    "logic": {
        "min_conditions": 2,         # Maintain quality
        "max_rules": 8,              # Balanced rules
        "confidence_threshold": 0.75, # Maintain quality
        "support_threshold": 0.25,   # Balanced threshold
        "consistency_weight": 0.6,   # Balanced weight
        "max_inconsistencies": 0.35  # Maintain quality
    }
}

# Large dataset configuration for datasets with 10,000+ rows
LARGE_DATASET_CONFIG = {
    "iterations": 8,   # Balanced iterations for large datasets
    "gamma": 0.6,      # Balanced for convergence and quality
    "alpha": 0.6,      # Balanced for convergence and quality
    "beta": 0.6,       # Balanced permanence processing
    "delta": 0.3,      # Balanced logic processing
    "initial_trust": 0.85,
    "initial_state": 0.80,
    "presence": {
        "entropy_threshold": 2.2,    # Maintain quality threshold
        "min_confidence": 0.30,      # Maintain quality threshold
        "entropy_penalty": 2.5,      # Balanced penalty
        "refinement_factor": 0.85    # Balanced refinement
    },
    "permanence": {
        "hash_algorithm": "md5",     # Faster hash algorithm
        "block_size": 2000,          # Large block size for speed
        "min_blocks": 1,             # Minimal blocks
        "max_deviation": 0.03,       # Maintain quality
        "convergence_threshold": 0.95, # Maintain quality
        "consistency_threshold": 0.75  # Maintain quality
    },
    "logic": {
        "min_conditions": 2,         # Maintain quality
        "max_rules": 8,              # Balanced rules
        "confidence_threshold": 0.75, # Maintain quality
        "support_threshold": 0.25,   # Balanced threshold
        "consistency_weight": 0.6,   # Balanced weight
        "max_inconsistencies": 0.35  # Maintain quality
    }
}

# Super-fast configuration for very small datasets (50 rows or less)
SUPER_FAST_CONFIG = {
    "iterations": 2,   # Minimal iterations for speed while maintaining quality
    "gamma": 0.6,      # Balanced for convergence and quality
    "alpha": 0.6,      # Balanced for convergence and quality
    "beta": 0.5,       # Balanced permanence processing
    "delta": 0.3,      # Balanced logic processing
    "initial_trust": 0.85,
    "initial_state": 0.80,
    "presence": {
        "entropy_threshold": 2.0,    # Maintain quality threshold
        "min_confidence": 0.30,      # Maintain quality threshold
        "entropy_penalty": 2.0,      # Balanced penalty
        "refinement_factor": 0.8     # Balanced refinement
    },
    "permanence": {
        "hash_algorithm": "md5",     # Faster hash algorithm
        "block_size": 50,            # Process all data at once
        "min_blocks": 1,             # Single block
        "max_deviation": 0.05,       # Maintain quality
        "convergence_threshold": 0.90, # Maintain quality
        "consistency_threshold": 0.70  # Maintain quality
    },
    "logic": {
        "min_conditions": 2,         # Maintain quality
        "max_rules": 5,              # Balanced rules
        "confidence_threshold": 0.70, # Maintain quality
        "support_threshold": 0.30,   # Balanced threshold
        "consistency_weight": 0.5,   # Balanced weight
        "max_inconsistencies": 0.40  # Maintain quality
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

# Industry/Sector specific configurations
INDUSTRY_SPECIFIC_CONFIG = {
    "health": {
        "name": "Health / Medical AI",
        "description": "Healthcare and medical applications",
        "accuracy_threshold": 0.98,  # High accuracy required for medical decisions
        "trust_threshold": 0.90,     # High trust for patient safety
        "entropy_threshold": 1.2,    # Low entropy for consistent predictions
        "max_blocks": 25,
        "consecutive_blocks_required": 2,
        "weights": {
            "accuracy": 0.45,  # Higher weight for medical accuracy
            "trust": 0.35,     # High weight for patient safety
            "entropy": 0.15,   # Lower weight but still important
            "block_count": 0.05 # Minimal weight for efficiency
        },
        "auto_refinement": True  # Enable auto-refinement for medical applications
    },
    "finance": {
        "name": "Financial Services",
        "description": "Banking, insurance, and financial applications",
        "accuracy_threshold": 0.95,  # High accuracy for financial decisions
        "trust_threshold": 0.88,     # High trust for compliance
        "entropy_threshold": 1.5,    # Moderate entropy tolerance
        "max_blocks": 20,
        "consecutive_blocks_required": 2,
        "weights": {
            "accuracy": 0.40,  # High weight for financial accuracy
            "trust": 0.40,     # Equal weight for compliance
            "entropy": 0.15,   # Moderate weight for consistency
            "block_count": 0.05 # Minimal weight for efficiency
        },
        "auto_refinement": True  # Enable auto-refinement for financial applications
    },
    "industrial": {
        "name": "Industrial / Manufacturing",
        "description": "Manufacturing and industrial applications",
        "accuracy_threshold": 0.92,  # Good accuracy for quality control
        "trust_threshold": 0.85,     # Reliable trust for production
        "entropy_threshold": 1.8,    # Higher entropy tolerance
        "max_blocks": 15,
        "consecutive_blocks_required": 2,
        "weights": {
            "accuracy": 0.35,  # Good weight for quality control
            "trust": 0.30,     # Moderate weight for reliability
            "entropy": 0.20,   # Higher weight for flexibility
            "block_count": 0.15 # Higher weight for efficiency
        },
        "auto_refinement": False  # Disable auto-refinement for industrial applications
    },
    "cybersecurity": {
        "name": "Cybersecurity",
        "description": "Security and threat detection applications",
        "accuracy_threshold": 0.96,  # Very high accuracy for security
        "trust_threshold": 0.92,     # Very high trust for security
        "entropy_threshold": 1.0,    # Very low entropy for precise detection
        "max_blocks": 30,
        "consecutive_blocks_required": 3,
        "weights": {
            "accuracy": 0.50,  # Highest weight for security accuracy
            "trust": 0.35,     # High weight for security trust
            "entropy": 0.10,   # Low weight but critical for precision
            "block_count": 0.05 # Minimal weight for efficiency
        },
        "auto_refinement": True  # Enable auto-refinement for security applications
    },
    "general": {
        "name": "General Purpose",
        "description": "General business and commercial applications",
        "accuracy_threshold": 0.90,  # Standard accuracy
        "trust_threshold": 0.82,     # Standard trust
        "entropy_threshold": 2.0,    # Standard entropy tolerance
        "max_blocks": 10,
        "consecutive_blocks_required": 2,
        "weights": {
            "accuracy": 0.40,  # Standard weight for accuracy
            "trust": 0.30,     # Standard weight for trust
            "entropy": 0.20,   # Standard weight for flexibility
            "block_count": 0.10 # Standard weight for efficiency
        },
        "auto_refinement": False  # Disable auto-refinement for general applications
    }
}

# Adaptive threshold evaluation configuration
ADAPTIVE_EVALUATION_CONFIG = {
    "enabled": True,
    "auto_refinement": True,
    "score_thresholds": {
        "excellent": 0.85,
        "acceptable": 0.70,
        "fail": 0.70
    },
    "soft_zones": {
        "accuracy_warning_zone": 0.95,  # 95% of threshold
        "trust_warning_zone": 0.95,     # 95% of threshold
        "entropy_warning_zone": 1.25,   # 125% of threshold
        "block_count_efficient": 0.8    # 80% of max blocks
    },
    "logging": {
        "save_evaluations": True,
        "log_level": "INFO",
        "output_format": "json"
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