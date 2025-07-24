"""
SREE Phase 1 Demo - Permanence Layer Validator
Hash-based logging for blockchain-inspired validation.
"""

import numpy as np
import hashlib
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

from layers.base import Validator
from config import PPP_CONFIG, LOGS_DIR


class PermanenceValidator(Validator):
    """
    Permanence Layer Validator - Hash-based logging for blockchain-inspired validation.
    
    This validator implements the blockchain-inspired component of the PPP loop,
    using hash-based logging to ensure data consistency and permanence. It's designed
    to provide immutable validation records and consistency checking.
    
    Target: Ensure data consistency and provide immutable validation records
    """
    
    def __init__(self, name: str = "PermanenceValidator", **kwargs):
        """
        Initialize Permanence validator with hash-based logging.
        
        Args:
            name: Validator name
            **kwargs: Additional configuration parameters
        """
        # Initialize state attributes first
        self._hash_algorithm = PPP_CONFIG["permanence"]["hash_algorithm"]
        self._block_size = PPP_CONFIG["permanence"]["block_size"]
        self._consistency_threshold = PPP_CONFIG["permanence"]["consistency_threshold"]
        self._ledger = []
        self._current_block = []
        self._block_counter = 0
        self._consistency_checks = []
        
        # Get permanence configuration
        permanence_config = PPP_CONFIG["permanence"].copy()
        permanence_config.update(kwargs)
        
        # Store configuration
        self._config = permanence_config
        
        # Call parent constructor last
        super().__init__(name=name)
    
    def validate(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Validate data using hash-based logging.
        
        Args:
            data: Input features (n_samples, n_features)
            labels: Optional ground truth labels for evaluation
            
        Returns:
            Trust scores based on consistency validation
        """
        logger = logging.getLogger(__name__)
        
        # Create validation records
        validation_records = self._create_validation_records(data, labels)
        
        # Add records to current block
        self._current_block.extend(validation_records)
        
        # Check if block is full and finalize
        if len(self._current_block) >= self._block_size:
            self._finalize_block()
        
        # Always finalize at least one block if we have records
        if len(self._current_block) > 0 and len(self._ledger) == 0:
            self._finalize_block()
        
        # Calculate consistency-based trust scores
        trust_scores = self._calculate_consistency_scores(data, validation_records)
        
        logger.info(f"Permanence processed {len(validation_records)} records, "
                   f"block size: {len(self._current_block)}/{self._block_size}, "
                   f"total blocks: {len(self._ledger)}")
        
        return trust_scores
    
    def _create_validation_records(self, data: np.ndarray, 
                                  labels: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Create validation records for the data.
        
        Args:
            data: Input features
            labels: Optional ground truth labels
            
        Returns:
            List of validation records
        """
        records = []
        timestamp = datetime.now().isoformat()
        
        for i, sample in enumerate(data):
            # Create hash of the sample
            sample_hash = self._hash_data(sample)
            
            # Create validation record
            record = {
                "index": i,
                "timestamp": timestamp,
                "sample_hash": sample_hash,
                "data_shape": sample.shape,
                "data_summary": {
                    "mean": float(np.mean(sample)),
                    "std": float(np.std(sample)),
                    "min": float(np.min(sample)),
                    "max": float(np.max(sample))
                }
            }
            
            # Add label if provided
            if labels is not None:
                record["label"] = int(labels[i])
            
            records.append(record)
        
        return records
    
    def _hash_data(self, data: np.ndarray) -> str:
        """
        Create hash of the data using hashlib.sha256.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash string
        """
        # Convert data to bytes with proper encoding
        data_bytes = data.tobytes()
        
        # Always use SHA256 for consistency and security
        hash_obj = hashlib.sha256(data_bytes)
        
        return hash_obj.hexdigest()
    
    def _finalize_block(self):
        """Finalize the current block and add it to the ledger."""
        if not self._current_block:
            return
        
        # Create block header
        block_header = {
            "block_number": self._block_counter,
            "timestamp": datetime.now().isoformat(),
            "record_count": len(self._current_block),
            "previous_hash": self._get_previous_hash()
        }
        
        # Create block hash
        block_data = json.dumps(self._current_block, sort_keys=True).encode()
        block_hash = hashlib.sha256(block_data).hexdigest()
        block_header["block_hash"] = block_hash
        
        # Create complete block
        block = {
            "header": block_header,
            "records": self._current_block
        }
        
        # Add to ledger
        self._ledger.append(block)
        
        # Reset current block
        self._current_block = []
        self._block_counter += 1
        
        logging.getLogger(__name__).info(f"Block {block_header['block_number']} finalized "
                                        f"with {block_header['record_count']} records")
    
    def _get_previous_hash(self) -> str:
        """Get hash of the previous block."""
        if not self._ledger:
            return "0" * 64  # Genesis block
        
        last_block = self._ledger[-1]
        return last_block["header"]["block_hash"]
    
    def _calculate_consistency_scores(self, data: np.ndarray, 
                                    records: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate consistency-based trust scores.
        
        Args:
            data: Input features
            records: Validation records
            
        Returns:
            Trust scores based on consistency
        """
        trust_scores = np.ones(len(data))
        
        # Check for duplicate samples (same hash)
        hash_counts = {}
        for record in records:
            sample_hash = record["sample_hash"]
            if sample_hash in hash_counts:
                hash_counts[sample_hash] += 1
            else:
                hash_counts[sample_hash] = 1
        
        # Reduce trust for duplicate samples
        for i, record in enumerate(records):
            sample_hash = record["sample_hash"]
            if hash_counts[sample_hash] > 1:
                # Duplicate detected - reduce trust
                trust_scores[i] *= 0.8
        
        # Check data consistency
        for i, record in enumerate(records):
            sample = data[i]
            summary = record["data_summary"]
            
            # Check if current sample matches recorded summary
            current_mean = np.mean(sample)
            current_std = np.std(sample)
            
            mean_diff = abs(current_mean - summary["mean"])
            std_diff = abs(current_std - summary["std"])
            
            # Reduce trust if data has changed significantly
            if mean_diff > 0.1 or std_diff > 0.1:
                trust_scores[i] *= 0.9
        
        # Store consistency check results
        self._consistency_checks.append({
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(data),
            "n_duplicates": sum(1 for count in hash_counts.values() if count > 1),
            "avg_trust": float(np.mean(trust_scores))
        })
        
        return trust_scores
    
    def check_ledger_consistency(self) -> Dict[str, Any]:
        """
        Check the consistency of the entire ledger.
        
        Returns:
            Consistency check results
        """
        if len(self._ledger) < 2:
            return {"status": "insufficient_blocks", "message": "Need at least 2 blocks"}
        
        inconsistencies = []
        
        for i in range(1, len(self._ledger)):
            current_block = self._ledger[i]
            previous_block = self._ledger[i-1]
            
            # Check if previous hash matches
            if current_block["header"]["previous_hash"] != previous_block["header"]["block_hash"]:
                inconsistencies.append({
                    "block_number": current_block["header"]["block_number"],
                    "issue": "previous_hash_mismatch"
                })
            
            # Verify block hash
            block_data = json.dumps(current_block["records"], sort_keys=True).encode()
            expected_hash = hashlib.sha256(block_data).hexdigest()
            
            if current_block["header"]["block_hash"] != expected_hash:
                inconsistencies.append({
                    "block_number": current_block["header"]["block_number"],
                    "issue": "block_hash_mismatch"
                })
        
        consistency_score = 1.0 - (len(inconsistencies) / len(self._ledger))
        
        return {
            "status": "complete",
            "total_blocks": len(self._ledger),
            "inconsistencies": inconsistencies,
            "consistency_score": consistency_score,
            "is_consistent": consistency_score >= self._consistency_threshold
        }
    
    def save_ledger(self, filename: str = "permanence_ledger.json") -> str:
        """
        Save the ledger to disk.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved ledger
        """
        # Finalize any remaining records
        if self._current_block:
            self._finalize_block()
        
        ledger_path = LOGS_DIR / filename
        
        # Save ledger as JSON
        with open(ledger_path, 'w') as f:
            json.dump(self._ledger, f, indent=2)
        
        logging.getLogger(__name__).info(f"Ledger saved to {ledger_path}")
        return str(ledger_path)
    
    def load_ledger(self, filename: str = "permanence_ledger.json") -> bool:
        """
        Load the ledger from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            True if loaded successfully
        """
        ledger_path = LOGS_DIR / filename
        
        if not ledger_path.exists():
            raise FileNotFoundError(f"Ledger file not found: {ledger_path}")
        
        with open(ledger_path, 'r') as f:
            self._ledger = json.load(f)
        
        # Update block counter
        self._block_counter = len(self._ledger)
        
        logging.getLogger(__name__).info(f"Ledger loaded from {ledger_path}")
        return True
    
    def get_ledger_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the ledger.
        
        Returns:
            Ledger statistics
        """
        if not self._ledger:
            return {"message": "No ledger data available"}
        
        total_records = sum(len(block["records"]) for block in self._ledger)
        
        stats = {
            "total_blocks": len(self._ledger),
            "total_records": total_records,
            "avg_records_per_block": total_records / len(self._ledger),
            "first_block_timestamp": self._ledger[0]["header"]["timestamp"],
            "last_block_timestamp": self._ledger[-1]["header"]["timestamp"],
            "consistency_checks": len(self._consistency_checks)
        }
        
        # Add consistency check results
        if self._consistency_checks:
            recent_checks = self._consistency_checks[-10:]  # Last 10 checks
            stats.update({
                "avg_trust_recent": np.mean([c["avg_trust"] for c in recent_checks]),
                "avg_duplicates_recent": np.mean([c["n_duplicates"] for c in recent_checks])
            })
        
        return stats
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get validator metadata including ledger information.
        
        Returns:
            Metadata dictionary
        """
        metadata = super().get_metadata()
        metadata.update({
            "hash_algorithm": self._hash_algorithm,
            "block_size": self._block_size,
            "consistency_threshold": self._consistency_threshold,
            "total_blocks": len(self._ledger),
            "current_block_size": len(self._current_block),
            "n_consistency_checks": len(self._consistency_checks),
            "description": "Hash-based logging for blockchain-inspired validation"
        })
        
        # Add ledger statistics if available
        ledger_stats = self.get_ledger_statistics()
        if "message" not in ledger_stats:
            metadata.update({
                "total_records": ledger_stats["total_records"],
                "avg_records_per_block": ledger_stats["avg_records_per_block"]
            })
        
        return metadata
    
    def reset(self):
        """Reset validator state."""
        self._ledger = []
        self._current_block = []
        self._block_counter = 0
        self._consistency_checks = []
    
    def get_state(self) -> Dict[str, Any]:
        """Get current validator state."""
        state = super().get_state()
        state.update({
            "hash_algorithm": self._hash_algorithm,
            "block_size": self._block_size,
            "consistency_threshold": self._consistency_threshold,
            "total_blocks": len(self._ledger),
            "current_block_size": len(self._current_block),
            "n_consistency_checks": len(self._consistency_checks)
        })
        return state
    
    def set_state(self, state: Dict[str, Any]):
        """Set validator state."""
        super().set_state(state)
        if "hash_algorithm" in state:
            self._hash_algorithm = state["hash_algorithm"]
        if "block_size" in state:
            self._block_size = state["block_size"]
        if "consistency_threshold" in state:
            self._consistency_threshold = state["consistency_threshold"]


def create_permanence_validator(**kwargs) -> PermanenceValidator:
    """
    Factory function to create a Permanence validator.
    
    Args:
        **kwargs: Configuration arguments
        
    Returns:
        Configured PermanenceValidator instance
    """
    return PermanenceValidator(**kwargs) 