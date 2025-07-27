#!/usr/bin/env python3
"""
Comprehensive New Heart Disease Dataset Analysis for SREE
Analyzes the real heart disease dataset with clinical features
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import SREE components
from data_loader import DataLoader
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from loop.trust_loop import TrustUpdateLoop
from config import setup_logging

class NewHeartDiseaseAnalyzer:
    def __init__(self):
        self.logger = setup_logging()
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_new_heart_disease_dataset(self):
        """Load and preprocess the new heart disease dataset"""
        self.logger.info("Loading new heart disease dataset...")
        
        # Load the new heart disease dataset
        df = pd.read_csv('heart_disease_dataset_new.csv')
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Store feature names
        self.feature_names = list(X.columns)
        self.target_names = ['No Heart Disease', 'Heart Disease']
        
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
        self.df = df
        
        # Store dataset info
        self.results['dataset_info'] = {
            'total_samples': len(df),
            'total_features': len(self.feature_names),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': {
                'class_0': int(np.sum(y == 0)),
                'class_1': int(np.sum(y == 1)),
                'balance_ratio': float(np.sum(y == 1) / len(y))
            },
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'dataset_source': 'Real Heart Disease Dataset with Clinical Features'
        }
        
        self.logger.info(f"New heart disease dataset loaded: {len(df)} samples, {len(self.feature_names)} features")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def analyze_preprocessing_steps(self):
        """Analyze and document preprocessing steps"""
        self.logger.info("Analyzing preprocessing steps...")
        
        # Calculate original statistics
        X_original = self.df.drop('target', axis=1)
        
        # Calculate scaling statistics
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_original)
        
        self.results['preprocessing_steps'] = {
            'scaling_method': {
                'method': 'StandardScaler',
                'description': 'Standardization (Z-score normalization)',
                'formula': 'z = (x - μ) / σ',
                'effect': 'Transforms features to have mean=0 and std=1',
                'before_scaling': {
                    'mean_range': [float(X_original.mean().min()), float(X_original.mean().max())],
                    'std_range': [float(X_original.std().min()), float(X_original.std().max())]
                },
                'after_scaling': {
                    'mean_range': [float(X_scaled.mean(axis=0).min()), float(X_scaled.mean(axis=0).max())],
                    'std_range': [float(X_scaled.std(axis=0).min()), float(X_scaled.std(axis=0).max())]
                }
            },
            'balancing_method': {
                'method': 'None (Original Distribution)',
                'description': 'No balancing applied - preserving natural medical dataset distribution',
                'reasoning': 'Medical datasets should preserve natural class distribution for clinical relevance',
                'class_distribution': self.results['dataset_info']['class_distribution']
            },
            'noise_injection': {
                'method': 'None',
                'description': 'No artificial noise added',
                'reasoning': 'Medical data should remain unmodified for clinical accuracy'
            },
            'outlier_handling': {
                'method': 'None (Clinical Data)',
                'description': 'No outlier handling applied - preserving clinical measurements',
                'reasoning': 'Clinical measurements should be preserved as they represent real patient data'
            },
            'feature_engineering': {
                'method': 'None',
                'description': 'No feature engineering applied',
                'reasoning': 'Using original clinical features for interpretability'
            }
        }
        
        self.logger.info("Preprocessing analysis complete")
    
    def generate_correlation_matrix(self):
        """Generate feature correlation matrix"""
        self.logger.info("Generating correlation matrix...")
        
        # Create DataFrame for correlation analysis
        df_analysis = self.df.copy()
        
        # Calculate correlation matrix
        corr_matrix = df_analysis.corr()
        
        # Store correlation data
        self.results['correlation_matrix'] = {
            'matrix_size': corr_matrix.shape,
            'correlation_range': {
                'min': float(corr_matrix.min().min()),
                'max': float(corr_matrix.max().max())
            },
            'target_correlations': corr_matrix['target'].sort_values(ascending=False).to_dict(),
            'high_correlations': {}
        }
        
        # Find high correlations (|r| > 0.5)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5 and i != j:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        self.results['correlation_matrix']['high_correlations'] = high_corr_pairs
        
        # Save correlation matrix
        corr_matrix.to_csv(f'logs/correlation_matrix_new_heart_{self.timestamp}.csv')
        
        # Generate heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix - New Heart Disease Dataset', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f'plots/correlation_matrix_new_heart_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Correlation matrix generated")
    
    def generate_class_balance_chart(self):
        """Generate class balance visualization"""
        self.logger.info("Generating class balance chart...")
        
        class_counts = pd.Series(self.y_train).value_counts()
        class_labels = [self.target_names[i] for i in class_counts.index]
        
        # Store class balance data
        self.results['class_balance'] = {
            'class_counts': class_counts.to_dict(),
            'class_labels': class_labels,
            'balance_ratio': float(class_counts[1] / class_counts[0]) if class_counts[0] > 0 else float('inf'),
            'imbalance_level': 'Balanced' if 0.8 <= class_counts[1] / class_counts[0] <= 1.2 else 'Imbalanced'
        }
        
        # Generate visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        bars = ax1.bar(class_labels, class_counts.values, color=['#ff9999', '#66b3ff'])
        ax1.set_title('Class Distribution - New Heart Disease Dataset', fontsize=14, pad=20)
        ax1.set_ylabel('Number of Samples')
        ax1.set_xlabel('Class')
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = ['#ff9999', '#66b3ff']
        wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax2.set_title('Class Balance Ratio', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(f'plots/class_balance_new_heart_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Class balance chart generated")
    
    def generate_variance_report(self):
        """Generate per-feature variance analysis"""
        self.logger.info("Generating variance report...")
        
        # Calculate variance for each feature
        variances = np.var(self.X_train, axis=0)
        feature_variance = pd.DataFrame({
            'feature': self.feature_names,
            'variance': variances,
            'std': np.std(self.X_train, axis=0),
            'mean': np.mean(self.X_train, axis=0),
            'min': np.min(self.X_train, axis=0),
            'max': np.max(self.X_train, axis=0)
        }).sort_values('variance', ascending=False)
        
        # Store variance data
        self.results['variance_report'] = {
            'total_features': len(self.feature_names),
            'variance_range': {
                'min': float(variances.min()),
                'max': float(variances.max()),
                'mean': float(variances.mean()),
                'std': float(variances.std())
            },
            'high_variance_features': feature_variance.head(10).to_dict('records'),
            'low_variance_features': feature_variance.tail(10).to_dict('records'),
            'variance_categories': {
                'high_variance': len(variances[variances > np.percentile(variances, 75)]),
                'medium_variance': len(variances[(variances >= np.percentile(variances, 25)) & 
                                               (variances <= np.percentile(variances, 75))]),
                'low_variance': len(variances[variances < np.percentile(variances, 25)])
            }
        }
        
        # Save variance data
        feature_variance.to_csv(f'logs/feature_variance_new_heart_{self.timestamp}.csv', index=False)
        
        # Generate visualization
        plt.figure(figsize=(15, 8))
        
        # Top 15 features by variance
        top_features = feature_variance.head(15)
        bars = plt.barh(range(len(top_features)), top_features['variance'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Variance')
        plt.title('Feature Variance Analysis - Top 15 Features', fontsize=14, pad=20)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, var) in enumerate(zip(bars, top_features['variance'])):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{var:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'plots/feature_variance_new_heart_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Variance report generated")
    
    def run_sree_analysis(self):
        """Run SREE analysis on new heart disease dataset"""
        self.logger.info("Running SREE analysis...")
        
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
        
        # Store SREE results
        self.results['sree_results'] = {
            'pattern_accuracy': float(pattern_results.get('test_accuracy', 0.0)),
            'final_accuracy': float(final_results['final_accuracy']),
            'final_trust': float(final_results['final_trust']),
            'convergence': bool(final_results['convergence_achieved']),
            'iterations': len(final_results['iterations']),
            'block_count': int(block_count),  # Use actual block count from permanence layer
            'entropy': float(entropy)  # Use actual entropy from presence layer
        }
        
        # Extract trust history
        trust_scores = []
        accuracies = []
        for t in trust_loop._trust_history:
            if isinstance(t, dict):
                trust_scores.append(float(t.get('mean_trust', 0.0)))
            else:
                trust_scores.append(float(t))
        
        for a in trust_loop._accuracy_history:
            if isinstance(a, dict):
                accuracies.append(float(a.get('accuracy', 0.0)))
            else:
                accuracies.append(float(a))
        
        self.results['trust_history'] = {
            'iterations': list(range(1, len(trust_loop._trust_history) + 1)),
            'trust_scores': trust_scores,
            'accuracies': accuracies
        }
        
        # Generate predictions for confusion matrix
        pattern_validator.validate(self.X_test)
        y_pred = pattern_validator.predictions
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        self.results['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1]),
            'accuracy': float((cm[0, 0] + cm[1, 1]) / cm.sum()),
            'precision': float(cm[1, 1] / (cm[1, 1] + cm[0, 1])) if (cm[1, 1] + cm[0, 1]) > 0 else 0.0,
            'recall': float(cm[1, 1] / (cm[1, 1] + cm[1, 0])) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
        }
        
        self.logger.info("SREE analysis complete")
    
    def generate_confusion_matrix_plot(self):
        """Generate confusion matrix visualization"""
        self.logger.info("Generating confusion matrix plot...")
        
        cm = np.array(self.results['confusion_matrix']['matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title('Confusion Matrix - New Heart Disease Dataset', fontsize=14, pad=20)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add metrics text
        metrics_text = f"Accuracy: {self.results['confusion_matrix']['accuracy']:.3f}\n"
        metrics_text += f"Precision: {self.results['confusion_matrix']['precision']:.3f}\n"
        metrics_text += f"Recall: {self.results['confusion_matrix']['recall']:.3f}"
        
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrix_new_heart_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Confusion matrix plot generated")
    
    def generate_trust_vs_iteration_plot(self):
        """Generate trust vs iteration plot"""
        self.logger.info("Generating trust vs iteration plot...")
        
        iterations = self.results['trust_history']['iterations']
        trust_scores = self.results['trust_history']['trust_scores']
        accuracies = self.results['trust_history']['accuracies']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Trust vs Iteration
        ax1.plot(iterations, trust_scores, 'b-o', linewidth=2, markersize=6, label='Trust Score')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Trust Score')
        ax1.set_title('Trust Score vs Iteration', fontsize=14, pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Accuracy vs Iteration
        ax2.plot(iterations, accuracies, 'r-s', linewidth=2, markersize=6, label='Accuracy')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Iteration', fontsize=14, pad=20)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/trust_vs_iteration_new_heart_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Trust vs iteration plot generated")
    
    def generate_comparison_summary(self):
        """Generate comparison summary with previous results"""
        self.logger.info("Generating comparison summary...")
        
        # Previous results from breast cancer dataset
        previous_results = {
            'accuracy': 0.9561,
            'trust_score': 0.9928,
            'block_count': 3,
            'entropy': 3.6,
            'dataset': 'UCI Breast Cancer (Proxy)'
        }
        
        # Current results
        current_results = {
            'accuracy': self.results['sree_results']['final_accuracy'],
            'trust_score': self.results['sree_results']['final_trust'],
            'block_count': self.results['sree_results']['block_count'],
            'entropy': self.results['sree_results']['entropy'],
            'iterations': self.results['sree_results']['iterations'],
            'dataset': 'Real Heart Disease Dataset'
        }
        
        self.results['comparison_summary'] = {
            'previous_results': previous_results,
            'current_results': current_results,
            'improvements': {
                'accuracy_change': current_results['accuracy'] - previous_results['accuracy'],
                'trust_change': current_results['trust_score'] - previous_results['trust_score'],
                'performance_status': 'Improved' if current_results['accuracy'] >= previous_results['accuracy'] else 'Declined'
            },
            'target_achievement': {
                'accuracy_target': 0.95,
                'trust_target': 0.85,
                'accuracy_achieved': current_results['accuracy'] >= 0.95,
                'trust_achieved': current_results['trust_score'] >= 0.85,
                'all_targets_met': (current_results['accuracy'] >= 0.95) and (current_results['trust_score'] >= 0.85)
            }
        }
        
        self.logger.info("Comparison summary generated")
    
    def save_results(self):
        """Save all results to files"""
        self.logger.info("Saving results...")
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        # Save JSON results
        with open(f'logs/new_heart_disease_analysis_{self.timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save comprehensive report
        report = f"""
============================================================
NEW HEART DISEASE DATASET COMPREHENSIVE ANALYSIS REPORT
============================================================
Timestamp: {self.timestamp}
Dataset: Real Heart Disease Dataset with Clinical Features
Total Samples: {self.results['dataset_info']['total_samples']}
Total Features: {self.results['dataset_info']['total_features']}

CLINICAL FEATURES:
{', '.join(self.feature_names)}

PREPROCESSING STEPS:
- Scaling Method: {self.results['preprocessing_steps']['scaling_method']['method']}
- Balancing Method: {self.results['preprocessing_steps']['balancing_method']['method']}
- Noise Injection: {self.results['preprocessing_steps']['noise_injection']['method']}

CLASS BALANCE:
- Class 0 ({self.target_names[0]}): {self.results['class_balance']['class_counts'][0]}
- Class 1 ({self.target_names[1]}): {self.results['class_balance']['class_counts'][1]}
- Balance Ratio: {self.results['class_balance']['balance_ratio']:.3f}
- Imbalance Level: {self.results['class_balance']['imbalance_level']}

SREE PERFORMANCE RESULTS:
- Final Accuracy: {self.results['sree_results']['final_accuracy']:.4f}
- Final Trust Score: {self.results['sree_results']['final_trust']:.4f}
- Convergence: {self.results['sree_results']['convergence']}
- Iterations: {self.results['sree_results']['iterations']}
- Block Count: {self.results['sree_results']['block_count']}
- Entropy: {self.results['sree_results']['entropy']:.4f}

CONFUSION MATRIX METRICS:
- True Negatives: {self.results['confusion_matrix']['true_negatives']}
- False Positives: {self.results['confusion_matrix']['false_positives']}
- False Negatives: {self.results['confusion_matrix']['false_negatives']}
- True Positives: {self.results['confusion_matrix']['true_positives']}
- Accuracy: {self.results['confusion_matrix']['accuracy']:.4f}
- Precision: {self.results['confusion_matrix']['precision']:.4f}
- Recall: {self.results['confusion_matrix']['recall']:.4f}

TARGET ACHIEVEMENT:
- Accuracy Target (≥95%): {'✅ ACHIEVED' if self.results['comparison_summary']['target_achievement']['accuracy_achieved'] else '❌ NOT ACHIEVED'}
- Trust Target (≥85%): {'✅ ACHIEVED' if self.results['comparison_summary']['target_achievement']['trust_achieved'] else '❌ NOT ACHIEVED'}
- All Targets Met: {'✅ YES' if self.results['comparison_summary']['target_achievement']['all_targets_met'] else '❌ NO'}

FILES GENERATED:
- JSON Results: logs/new_heart_disease_analysis_{self.timestamp}.json
- Correlation Matrix: plots/correlation_matrix_new_heart_{self.timestamp}.png
- Class Balance: plots/class_balance_new_heart_{self.timestamp}.png
- Feature Variance: plots/feature_variance_new_heart_{self.timestamp}.png
- Confusion Matrix: plots/confusion_matrix_new_heart_{self.timestamp}.png
- Trust vs Iteration: plots/trust_vs_iteration_new_heart_{self.timestamp}.png
============================================================
"""
        
        with open(f'logs/new_heart_disease_analysis_report_{self.timestamp}.txt', 'w') as f:
            f.write(report)
        
        self.logger.info("Results saved successfully")
    
    def run_complete_analysis(self):
        """Run complete new heart disease analysis"""
        self.logger.info("Starting comprehensive new heart disease analysis...")
        
        # Load and analyze dataset
        self.load_new_heart_disease_dataset()
        self.analyze_preprocessing_steps()
        
        # Generate visualizations
        self.generate_correlation_matrix()
        self.generate_class_balance_chart()
        self.generate_variance_report()
        
        # Run SREE analysis
        self.run_sree_analysis()
        
        # Generate additional visualizations
        self.generate_confusion_matrix_plot()
        self.generate_trust_vs_iteration_plot()
        
        # Generate comparison summary
        self.generate_comparison_summary()
        
        # Save all results
        self.save_results()
        
        self.logger.info("Comprehensive new heart disease analysis complete!")
        
        # Print summary
        print("\n" + "="*60)
        print("NEW HEART DISEASE DATASET ANALYSIS COMPLETE")
        print("="*60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Dataset: Real Heart Disease Dataset")
        print(f"Features: {len(self.feature_names)} clinical features")
        print(f"Accuracy: {self.results['sree_results']['final_accuracy']:.4f}")
        print(f"Trust Score: {self.results['sree_results']['final_trust']:.4f}")
        print(f"Convergence: {self.results['sree_results']['convergence']}")
        print(f"Block Count: {self.results['sree_results']['block_count']}")
        print("\nFiles generated:")
        print(f"- logs/new_heart_disease_analysis_{self.timestamp}.json")
        print(f"- logs/new_heart_disease_analysis_report_{self.timestamp}.txt")
        print(f"- plots/correlation_matrix_new_heart_{self.timestamp}.png")
        print(f"- plots/class_balance_new_heart_{self.timestamp}.png")
        print(f"- plots/feature_variance_new_heart_{self.timestamp}.png")
        print(f"- plots/confusion_matrix_new_heart_{self.timestamp}.png")
        print(f"- plots/trust_vs_iteration_new_heart_{self.timestamp}.png")
        print("="*60)

if __name__ == "__main__":
    analyzer = NewHeartDiseaseAnalyzer()
    analyzer.run_complete_analysis() 