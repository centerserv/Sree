"""
SREE Phase 1 Demo - Base Validator Interface
Base class for all PPP validators with Phase 2 hooks for Qiskit/Ganache integration.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple


class Validator(ABC):
    """
    Base class for all PPP validators with Phase 2 hooks.
    
    This abstract base class defines the interface that all Pattern, Presence,
    Permanence, and Logic validators must implement. It's designed to be
    modular for future Phase 2 integration with Qiskit (quantum) and
    Ganache (blockchain).
    """
    
    def __init__(self, name: str = None):
        """
        Initialize validator with optional name.
        
        Args:
            name: Optional name for the validator (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.metadata = self.get_metadata()
    
    @abstractmethod
    def validate(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Validate input data and return trust scores.
        
        Args:
            data: Input data to validate (shape depends on validator type)
            labels: Optional ground truth labels for supervised validation
            
        Returns:
            Trust scores as numpy array (values typically in [0, 1])
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return validator metadata for logging and Phase 2 integration.
        
        Returns:
            Dictionary containing validator metadata
        """
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "phase": 1,  # Phase 1 simulation
            "description": self.__doc__ or "SREE PPP Validator"
        }
    
    def reset(self):
        """
        Reset validator state (useful for testing and Phase 2).
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current validator state for Phase 2 integration.
        
        Returns:
            Dictionary containing current state
        """
        return {
            "name": self.name,
            "metadata": self.metadata
        }
    
    def set_state(self, state: Dict[str, Any]):
        """
        Set validator state from Phase 2 integration.
        
        Args:
            state: Dictionary containing state to restore
        """
        if "name" in state:
            self.name = state["name"]
        if "metadata" in state:
            self.metadata.update(state["metadata"])
    
    def __str__(self) -> str:
        """String representation of the validator."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the validator."""
        return f"{self.__class__.__name__}(name='{self.name}', metadata={self.metadata})" 