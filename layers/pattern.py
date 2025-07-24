"""
SREE Phase 1 Demo - Pattern Layer Validator
MLP classifier for pattern recognition (AI component).
"""

import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from layers.base import Validator
from config import MODEL_CONFIG, MODELS_DIR


class PatternValidator(Validator):
    """
    Pattern Layer Validator - MLP classifier for pattern recognition.
    
    This validator implements the AI component of the PPP loop, using a
    Multi-Layer Perceptron (MLP) classifier to achieve baseline ~85% accuracy
    on MNIST dataset. It's designed to be modular for Phase 2 integration.
    
    Target: ~85% accuracy (baseline from manuscript Table 3)
    """
    
    def __init__(self, name: str = "PatternValidator", **kwargs):
        """
        Initialize Pattern validator with MLP classifier.
        
        Args:
            name: Validator name
            **kwargs: Additional arguments for MLP configuration
        """
        # Initialize state attributes first (before calling super().__init__)
        self._is_trained = False
        self._accuracy = 0.0
        self._predictions = None
        self._probabilities = None
        self._training_history = []
        self._validation_scores = []
        
        # Get MLP configuration
        mlp_config = MODEL_CONFIG["mlp"].copy()
        mlp_config.update(kwargs)
        
        # Initialize MLP classifier
        self.model = MLPClassifier(**mlp_config)
        
        # Call parent constructor last
        super().__init__(name=name)
    
    @property
    def is_trained(self):
        """Check if model is trained."""
        return self._is_trained
    
    @property
    def accuracy(self):
        """Get current accuracy."""
        return self._accuracy
    
    @property
    def predictions(self):
        """Get current predictions."""
        return self._predictions
    
    @property
    def probabilities(self):
        """Get current probabilities."""
        return self._probabilities
    
    @property
    def training_history(self):
        """Get training history."""
        return self._training_history
        
    def validate(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Validate data using the trained MLP classifier.
        
        Args:
            data: Input features (n_samples, n_features)
            labels: Optional ground truth labels for evaluation
            
        Returns:
            Trust scores based on prediction confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before validation")
        
        # Get predictions and probabilities
        self._predictions = self.model.predict(data)
        self._probabilities = self.model.predict_proba(data)
        
        # Calculate trust scores based on prediction confidence
        # Higher confidence = higher trust score
        trust_scores = np.max(self._probabilities, axis=1)
        
        # Evaluate accuracy if labels provided
        if labels is not None:
            self._accuracy = accuracy_score(labels, self._predictions)
            logging.getLogger(__name__).info(f"Pattern accuracy: {self._accuracy:.4f}")
        
        return trust_scores
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input features (n_samples, n_features)
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(data)
    
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities on new data.
        
        Args:
            data: Input features (n_samples, n_features)
            
        Returns:
            Probability array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(data)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the MLP classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Training results dictionary
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Training Pattern validator with {len(X_train)} samples")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self._is_trained = True
        
        # Evaluate on training data
        train_accuracy = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Evaluate on validation data if provided
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_accuracy = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Perform cross-validation with appropriate number of folds
        n_samples = len(X_train)
        n_classes = len(np.unique(y_train))
        
        # Use fewer folds for small datasets to avoid warnings
        if n_samples < 50 or n_classes < 2:
            cv_folds = 2
        elif n_samples < 100:
            cv_folds = 3
        else:
            cv_folds = 5
            
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        logger.info(f"Cross-validation ({cv_folds}-fold): {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Store results
        results = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_scores": cv_scores.tolist(),
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train))
        }
        
        self._training_history.append(results)
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation results dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger = logging.getLogger(__name__)
        logger.info(f"Evaluating Pattern validator on {len(X_test)} test samples")
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        # Calculate metrics
        # Ensure proper formatting for binary classification
        y_test_formatted = np.array(y_test).astype(int)
        predictions_formatted = np.array(predictions).astype(int)
        
        accuracy = accuracy_score(y_test_formatted, predictions_formatted)
        conf_matrix = confusion_matrix(y_test_formatted, predictions_formatted)
        
        # Calculate per-class accuracy
        class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        # Calculate confidence-based trust scores
        trust_scores = np.max(probabilities, axis=1)
        avg_trust = np.mean(trust_scores)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Average trust score: {avg_trust:.4f}")
        
        results = {
            "accuracy": accuracy,
            "trust_scores": trust_scores.tolist(),
            "avg_trust": avg_trust,
            "confusion_matrix": conf_matrix.tolist(),
            "class_accuracy": class_accuracy.tolist(),
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
        
        return results
    
    def save_model(self, filename: str = "pattern_validator.joblib") -> str:
        """
        Save the trained model to disk.
        
        Args:
            filename: Model filename
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_path = MODELS_DIR / filename
        model_data = {
            "model": self.model,
            "is_trained": self._is_trained,
            "accuracy": self._accuracy,
            "training_history": self._training_history,
            "metadata": self.get_metadata()
        }
        
        joblib.dump(model_data, model_path)
        logging.getLogger(__name__).info(f"Pattern model saved to {model_path}")
        
        return str(model_path)
    
    def load_model(self, filename: str = "pattern_validator.joblib") -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filename: Model filename
            
        Returns:
            True if loaded successfully
        """
        model_path = MODELS_DIR / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data["model"]
        self._is_trained = model_data["is_trained"]
        self._accuracy = model_data.get("accuracy", 0.0)
        self._training_history = model_data.get("training_history", [])
        
        logging.getLogger(__name__).info(f"Pattern model loaded from {model_path}")
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get validator metadata including model information.
        
        Returns:
            Metadata dictionary
        """
        metadata = super().get_metadata()
        metadata.update({
            "model_type": "MLPClassifier",
            "is_trained": self._is_trained,
            "accuracy": self._accuracy,
            "n_training_runs": len(self._training_history),
            "target_accuracy": 0.85,  # Baseline from manuscript
            "description": "MLP classifier for pattern recognition (AI component)"
        })
        
        if self.is_trained:
            metadata.update({
                "n_features": self.model.n_features_in_,
                "n_classes": len(self.model.classes_),
                "hidden_layer_sizes": self.model.hidden_layer_sizes
            })
        
        return metadata
    
    def reset(self):
        """Reset validator state."""
        self.model = MLPClassifier(**MODEL_CONFIG["mlp"])
        self._is_trained = False
        self._accuracy = 0.0
        self._predictions = None
        self._probabilities = None
        self._training_history = []
        self._validation_scores = []
    
    def get_state(self) -> Dict[str, Any]:
        """Get current validator state."""
        state = super().get_state()
        state.update({
            "is_trained": self._is_trained,
            "accuracy": self._accuracy,
            "n_training_runs": len(self._training_history)
        })
        return state
    
    def set_state(self, state: Dict[str, Any]):
        """Set validator state."""
        super().set_state(state)
        if "is_trained" in state:
            self._is_trained = state["is_trained"]
        if "accuracy" in state:
            self._accuracy = state["accuracy"]


def create_pattern_validator(**kwargs) -> PatternValidator:
    """
    Factory function to create a Pattern validator.
    
    Args:
        **kwargs: Configuration arguments
        
    Returns:
        Configured PatternValidator instance
    """
    return PatternValidator(**kwargs) 