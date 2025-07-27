#!/usr/bin/env python3
"""
Debug Block Count Differences
Investigates why block counts differ between local and remote environments
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Import SREE components
from data_loader import DataLoader
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from loop.trust_loop import TrustUpdateLoop
from config import setup_logging, PPP_CONFIG

class BlockCountDebugger:
    def __init__(self):
        self.logger = setup_logging()
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_heart_disease_dataset(self):
        """Load heart disease dataset for analysis"""
        self.logger.info("Loading heart disease dataset for block count debugging...")
        
        # Load the heart disease dataset
        df = pd.read_csv('heart_disease_dataset_new.csv')
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        self.logger.info(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def analyze_permanence_block_creation(self):
        """Analyze permanence layer block creation logic"""
        self.logger.info("Analyzing permanence layer block creation...")
        
        # Initialize permanence validator
        permanence_validator = PermanenceValidator()
        
        # Get configuration
        block_size = PPP_CONFIG["permanence"]["block_size"]
        consistency_threshold = PPP_CONFIG["permanence"]["consistency_threshold"]
        
        self.logger.info(f"Block size: {block_size}")
        self.logger.info(f"Consistency threshold: {consistency_threshold}")
        
        # Analyze test data
        validation_records = permanence_validator._create_validation_records(self.X_test, self.y_test)
        
        # Calculate trust scores
        trust_scores = permanence_validator._calculate_consistency_scores(self.X_test, validation_records)
        
        # Analyze trust score distribution
        trust_stats = {
            'mean': float(np.mean(trust_scores)),
            'std': float(np.std(trust_scores)),
            'min': float(np.min(trust_scores)),
            'max': float(np.max(trust_scores)),
            'percentile_25': float(np.percentile(trust_scores, 25)),
            'percentile_75': float(np.percentile(trust_scores, 75)),
            'total_samples': len(trust_scores)
        }
        
        # Analyze block creation conditions
        high_confidence_mask = trust_scores > np.percentile(trust_scores, 75)
        low_confidence_mask = trust_scores < np.percentile(trust_scores, 25)
        
        block_conditions = {
            'high_confidence_count': int(np.sum(high_confidence_mask)),
            'low_confidence_count': int(np.sum(low_confidence_mask)),
            'high_confidence_threshold': block_size // 2,  # 20
            'low_confidence_threshold': block_size // 3,   # ~13
            'high_confidence_trigger': np.sum(high_confidence_mask) >= block_size // 2,
            'low_confidence_trigger': np.sum(low_confidence_mask) >= block_size // 3,
            'total_records': len(validation_records)
        }
        
        self.results['permanence_analysis'] = {
            'configuration': {
                'block_size': block_size,
                'consistency_threshold': consistency_threshold
            },
            'trust_score_distribution': trust_stats,
            'block_creation_conditions': block_conditions,
            'validation_records_count': len(validation_records)
        }
        
        self.logger.info(f"Trust score mean: {trust_stats['mean']:.4f}")
        self.logger.info(f"High confidence samples: {block_conditions['high_confidence_count']}")
        self.logger.info(f"Low confidence samples: {block_conditions['low_confidence_count']}")
        
        return trust_scores, validation_records
    
    def simulate_block_creation(self, trust_scores, validation_records):
        """Simulate the block creation process step by step"""
        self.logger.info("Simulating block creation process...")
        
        # Initialize permanence validator
        permanence_validator = PermanenceValidator()
        
        # Reset to clean state
        permanence_validator.reset()
        
        # Simulate the validate method step by step
        block_creation_log = []
        
        # Step 1: Create validation records
        records = validation_records.copy()
        current_block = records.copy()
        
        # Step 2: Calculate trust scores
        scores = trust_scores.copy()
        
        # Step 3: Analyze trust score distribution
        high_confidence_mask = scores > np.percentile(scores, 75)
        low_confidence_mask = scores < np.percentile(scores, 25)
        
        block_size = PPP_CONFIG["permanence"]["block_size"]
        
        # Step 4: Check high confidence condition
        high_conf_count = np.sum(high_confidence_mask)
        high_conf_trigger = high_conf_count >= block_size // 2
        
        block_creation_log.append({
            'step': 'high_confidence_check',
            'high_confidence_count': int(high_conf_count),
            'threshold': block_size // 2,
            'triggered': high_conf_trigger
        })
        
        # Step 5: Check low confidence condition
        low_conf_count = np.sum(low_confidence_mask)
        low_conf_trigger = low_conf_count >= block_size // 3
        
        block_creation_log.append({
            'step': 'low_confidence_check',
            'low_confidence_count': int(low_conf_count),
            'threshold': block_size // 3,
            'triggered': low_conf_trigger
        })
        
        # Step 6: Simulate block creation decisions
        blocks_created = 0
        
        # Always create at least one block
        if len(current_block) > 0:
            blocks_created += 1
            block_creation_log.append({
                'step': 'minimum_block',
                'reason': 'Always create at least one block',
                'records_count': len(current_block)
            })
        
        # Create block for high confidence samples
        if high_conf_trigger:
            blocks_created += 1
            block_creation_log.append({
                'step': 'high_confidence_block',
                'reason': f'High confidence samples ({high_conf_count}) >= threshold ({block_size // 2})',
                'records_count': int(high_conf_count)
            })
        
        # Create separate block for low confidence samples
        if low_conf_trigger:
            blocks_created += 1
            block_creation_log.append({
                'step': 'low_confidence_block',
                'reason': f'Low confidence samples ({low_conf_count}) >= threshold ({block_size // 3})',
                'records_count': int(low_conf_count)
            })
        
        # Ensure minimum 3 blocks for convergence
        if blocks_created < 3 and len(current_block) >= block_size // 2:
            blocks_created = 3
            block_creation_log.append({
                'step': 'convergence_blocks',
                'reason': 'Ensure minimum 3 blocks for convergence',
                'records_count': len(current_block)
            })
        
        self.results['block_simulation'] = {
            'total_blocks_created': blocks_created,
            'block_creation_log': block_creation_log,
            'final_block_count': max(blocks_created, 1)  # At least 1 block
        }
        
        self.logger.info(f"Simulated block creation: {blocks_created} blocks")
        
        return blocks_created, block_creation_log
    
    def run_full_sree_analysis(self):
        """Run full SREE analysis to see actual block count"""
        self.logger.info("Running full SREE analysis...")
        
        # Initialize SREE components
        pattern_validator = PatternValidator()
        presence_validator = PresenceValidator()
        permanence_validator = PermanenceValidator()
        logic_validator = LogicValidator()
        
        # Train pattern validator
        pattern_results = pattern_validator.train(self.X_train, self.y_train, self.X_test, self.y_test)
        
        # Run trust update loop
        trust_loop = TrustUpdateLoop(
            validators=[pattern_validator, presence_validator, permanence_validator, logic_validator]
        )
        
        # Run PPP loop
        final_results = trust_loop.run_ppp_loop(self.X_train, self.y_train, self.X_test, self.y_test)
        
        # Get permanence validator state
        permanence_state = permanence_validator.get_state()
        ledger_stats = permanence_validator.get_ledger_statistics()
        
        self.results['full_sree_analysis'] = {
            'final_accuracy': float(final_results['final_accuracy']),
            'final_trust': float(final_results['final_trust']),
            'convergence': bool(final_results['convergence_achieved']),
            'iterations': len(final_results['iterations']),
            'permanence_state': permanence_state,
            'ledger_statistics': ledger_stats
        }
        
        self.logger.info(f"Full SREE analysis complete")
        self.logger.info(f"Final accuracy: {final_results['final_accuracy']:.4f}")
        self.logger.info(f"Final trust: {final_results['final_trust']:.4f}")
        self.logger.info(f"Block count: {ledger_stats.get('total_blocks', 0)}")
        
        return final_results, ledger_stats
    
    def save_debug_results(self):
        """Save debug results to file"""
        self.logger.info("Saving debug results...")
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Save JSON results
        with open(f'logs/block_count_debug_{self.timestamp}.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, bool):
                    return bool(obj)
                elif hasattr(obj, 'isoformat'):  # datetime objects
                    return obj.isoformat()
                else:
                    return str(obj)  # Convert any other types to string
            
            serializable_results = convert_numpy_types(self.results)
            json.dump(serializable_results, f, indent=2)
        
        # Save text report
        report = f"""
============================================================
BLOCK COUNT DEBUG ANALYSIS REPORT
============================================================
Timestamp: {self.timestamp}

CONFIGURATION:
- Block Size: {self.results['permanence_analysis']['configuration']['block_size']}
- Consistency Threshold: {self.results['permanence_analysis']['configuration']['consistency_threshold']}

TRUST SCORE DISTRIBUTION:
- Mean: {self.results['permanence_analysis']['trust_score_distribution']['mean']:.4f}
- Std: {self.results['permanence_analysis']['trust_score_distribution']['std']:.4f}
- Min: {self.results['permanence_analysis']['trust_score_distribution']['min']:.4f}
- Max: {self.results['permanence_analysis']['trust_score_distribution']['max']:.4f}
- 25th Percentile: {self.results['permanence_analysis']['trust_score_distribution']['percentile_25']:.4f}
- 75th Percentile: {self.results['permanence_analysis']['trust_score_distribution']['percentile_75']:.4f}

BLOCK CREATION CONDITIONS:
- High Confidence Count: {self.results['permanence_analysis']['block_creation_conditions']['high_confidence_count']}
- Low Confidence Count: {self.results['permanence_analysis']['block_creation_conditions']['low_confidence_count']}
- High Confidence Threshold: {self.results['permanence_analysis']['block_creation_conditions']['high_confidence_threshold']}
- Low Confidence Threshold: {self.results['permanence_analysis']['block_creation_conditions']['low_confidence_threshold']}
- High Confidence Triggered: {self.results['permanence_analysis']['block_creation_conditions']['high_confidence_trigger']}
- Low Confidence Triggered: {self.results['permanence_analysis']['block_creation_conditions']['low_confidence_trigger']}

BLOCK SIMULATION:
- Simulated Blocks: {self.results['block_simulation']['total_blocks_created']}
- Final Block Count: {self.results['block_simulation']['final_block_count']}

FULL SREE ANALYSIS:
- Final Accuracy: {self.results['full_sree_analysis']['final_accuracy']:.4f}
- Final Trust: {self.results['full_sree_analysis']['final_trust']:.4f}
- Convergence: {self.results['full_sree_analysis']['convergence']}
- Iterations: {self.results['full_sree_analysis']['iterations']}
- Actual Block Count: {self.results['full_sree_analysis']['ledger_statistics'].get('total_blocks', 0)}

POTENTIAL CAUSES FOR BLOCK COUNT DIFFERENCES:
1. Trust Score Distribution: Different trust score distributions between environments
2. Random State: Different random seeds affecting model initialization
3. Data Processing: Different execution order or memory constraints
4. Convergence Speed: Different convergence patterns affecting block creation
5. Environment Differences: CPU, memory, or library version differences

RECOMMENDATIONS:
1. Set fixed random seeds for reproducible results
2. Log trust score distributions for comparison
3. Implement deterministic block creation logic
4. Add environment information logging
5. Consider making block creation more predictable
============================================================
"""
        
        with open(f'logs/block_count_debug_report_{self.timestamp}.txt', 'w') as f:
            f.write(report)
        
        self.logger.info("Debug results saved successfully")
    
    def run_complete_debug(self):
        """Run complete block count debugging analysis"""
        self.logger.info("Starting block count debugging analysis...")
        
        # Load dataset
        self.load_heart_disease_dataset()
        
        # Analyze permanence block creation
        trust_scores, validation_records = self.analyze_permanence_block_creation()
        
        # Simulate block creation
        simulated_blocks, block_log = self.simulate_block_creation(trust_scores, validation_records)
        
        # Run full SREE analysis
        final_results, ledger_stats = self.run_full_sree_analysis()
        
        # Save results
        self.save_debug_results()
        
        self.logger.info("Block count debugging analysis complete!")
        
        # Print summary
        print("\n" + "="*60)
        print("BLOCK COUNT DEBUG ANALYSIS COMPLETE")
        print("="*60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Simulated Blocks: {simulated_blocks}")
        print(f"Actual Blocks: {ledger_stats.get('total_blocks', 0)}")
        print(f"Trust Score Mean: {self.results['permanence_analysis']['trust_score_distribution']['mean']:.4f}")
        print(f"High Confidence Samples: {self.results['permanence_analysis']['block_creation_conditions']['high_confidence_count']}")
        print(f"Low Confidence Samples: {self.results['permanence_analysis']['block_creation_conditions']['low_confidence_count']}")
        print("\nFiles generated:")
        print(f"- logs/block_count_debug_{self.timestamp}.json")
        print(f"- logs/block_count_debug_report_{self.timestamp}.txt")
        print("="*60)

if __name__ == "__main__":
    debugger = BlockCountDebugger()
    debugger.run_complete_debug() 