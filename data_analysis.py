"""
SREE Phase 1 Demo - Dataset Analysis
Generate and save dataset analysis including correlation matrix, class distribution, and feature summary.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any

from config import DATA_DIR, PLOTS_DIR


def analyze_credit_dataset(dataset_path: str = "data/synthetic_credit_risk.csv") -> Dict[str, Any]:
    """
    Analyze the credit risk dataset and generate comprehensive report.
    
    Args:
        dataset_path: Path to the credit risk dataset CSV file
        
    Returns:
        Dictionary containing analysis results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing credit dataset: {dataset_path}")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Basic dataset info
    dataset_info = {
        "total_samples": len(df),
        "total_features": len(df.columns) - 1,  # Exclude target
        "feature_names": list(df.columns[:-1]),
        "target_name": df.columns[-1],
        "missing_values": df.isnull().sum().to_dict()
    }
    
    # Class distribution
    class_distribution = df['target'].value_counts().to_dict()
    class_balance = {
        "class_0_count": class_distribution.get(0, 0),
        "class_1_count": class_distribution.get(1, 0),
        "class_0_percentage": (class_distribution.get(0, 0) / len(df)) * 100,
        "class_1_percentage": (class_distribution.get(1, 0) / len(df)) * 100,
        "is_balanced": abs(class_distribution.get(0, 0) - class_distribution.get(1, 0)) <= 10
    }
    
    # Feature summary statistics
    feature_summary = df.describe().to_dict()
    
    # Correlation matrix
    correlation_matrix = df.corr().to_dict()
    
    # Feature importance (correlation with target)
    target_correlations = df.corr()['target'].abs().sort_values(ascending=False)
    feature_importance = target_correlations.to_dict()
    
    # Identify potential outliers (using IQR method)
    outliers = {}
    for column in df.columns[:-1]:  # Exclude target
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
        outliers[column] = {
            "count": outlier_count,
            "percentage": (outlier_count / len(df)) * 100
        }
    
    # Data quality assessment
    data_quality = {
        "has_missing_values": df.isnull().sum().sum() > 0,
        "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100
    }
    
    # Compile results
    analysis_results = {
        "dataset_info": dataset_info,
        "class_distribution": class_distribution,
        "class_balance": class_balance,
        "feature_summary": feature_summary,
        "correlation_matrix": correlation_matrix,
        "feature_importance": feature_importance,
        "outliers": outliers,
        "data_quality": data_quality
    }
    
    logger.info(f"Dataset analysis completed. Found {len(df)} samples with {len(df.columns)-1} features")
    logger.info(f"Class distribution: {class_distribution}")
    
    return analysis_results


def generate_correlation_matrix_plot(correlation_matrix: Dict[str, Any], 
                                   save_path: str = "plots/correlation_matrix.png") -> str:
    """
    Generate and save correlation matrix heatmap.
    
    Args:
        correlation_matrix: Correlation matrix dictionary
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    # Convert dictionary to DataFrame
    corr_df = pd.DataFrame(correlation_matrix)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def generate_class_distribution_plot(class_distribution: Dict[str, Any], 
                                   save_path: str = "plots/class_distribution.png") -> str:
    """
    Generate and save class distribution plot.
    
    Args:
        class_distribution: Class distribution dictionary
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    # Create class distribution plot
    plt.figure(figsize=(10, 6))
    
    # Bar plot
    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())
    colors = ['#ff7f0e', '#1f77b4']
    
    bars = plt.bar(classes, counts, color=colors, alpha=0.7)
    plt.title('Class Distribution', fontsize=16, pad=20)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Add percentage labels
    total = sum(counts)
    for i, (class_name, count) in enumerate(zip(classes, counts)):
        percentage = (count / total) * 100
        plt.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center', 
                fontweight='bold', color='white')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def generate_feature_importance_plot(feature_importance: Dict[str, Any], 
                                   save_path: str = "plots/feature_importance.png") -> str:
    """
    Generate and save feature importance plot.
    
    Args:
        feature_importance: Feature importance dictionary
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    # Remove target from feature importance
    feature_importance = {k: v for k, v in feature_importance.items() if k != 'target'}
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_features)
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(features, importance, color='skyblue', alpha=0.7)
    plt.title('Feature Importance (Correlation with Target)', fontsize=16, pad=20)
    plt.xlabel('Absolute Correlation with Target', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Add value labels on bars
    for bar, imp in zip(bars, importance):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def save_analysis_report(analysis_results: Dict[str, Any], 
                        save_path: str = "data/credit_analysis_report.txt") -> str:
    """
    Save comprehensive analysis report to text file.
    
    Args:
        analysis_results: Analysis results dictionary
        save_path: Path to save the report
        
    Returns:
        Path to saved report
    """
    with open(save_path, 'w') as f:
        f.write("CREDIT RISK DATASET ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total samples: {analysis_results['dataset_info']['total_samples']}\n")
        f.write(f"Total features: {analysis_results['dataset_info']['total_features']}\n")
        f.write(f"Feature names: {', '.join(analysis_results['dataset_info']['feature_names'])}\n")
        f.write(f"Target name: {analysis_results['dataset_info']['target_name']}\n\n")
        
        # Class distribution
        f.write("CLASS DISTRIBUTION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Class 0: {analysis_results['class_distribution'].get(0, 0)} samples ({analysis_results['class_balance']['class_0_percentage']:.1f}%)\n")
        f.write(f"Class 1: {analysis_results['class_distribution'].get(1, 0)} samples ({analysis_results['class_balance']['class_1_percentage']:.1f}%)\n")
        f.write(f"Balanced dataset: {'Yes' if analysis_results['class_balance']['is_balanced'] else 'No'}\n\n")
        
        # Feature importance
        f.write("FEATURE IMPORTANCE (Correlation with Target):\n")
        f.write("-" * 40 + "\n")
        feature_importance = {k: v for k, v in analysis_results['feature_importance'].items() if k != 'target'}
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            f.write(f"{feature}: {importance:.4f}\n")
        f.write("\n")
        
        # Data quality
        f.write("DATA QUALITY ASSESSMENT:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Missing values: {'Yes' if analysis_results['data_quality']['has_missing_values'] else 'No'}\n")
        f.write(f"Missing percentage: {analysis_results['data_quality']['missing_percentage']:.2f}%\n")
        f.write(f"Duplicate rows: {analysis_results['data_quality']['duplicate_rows']}\n")
        f.write(f"Duplicate percentage: {analysis_results['data_quality']['duplicate_percentage']:.2f}%\n\n")
        
        # Outliers summary
        f.write("OUTLIERS SUMMARY:\n")
        f.write("-" * 15 + "\n")
        total_outliers = sum(outlier['count'] for outlier in analysis_results['outliers'].values())
        f.write(f"Total outliers detected: {total_outliers}\n")
        f.write("Outliers by feature:\n")
        for feature, outlier_info in analysis_results['outliers'].items():
            f.write(f"  {feature}: {outlier_info['count']} ({outlier_info['percentage']:.1f}%)\n")
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 15 + "\n")
        if analysis_results['data_quality']['missing_percentage'] > 5:
            f.write("- Consider handling missing values\n")
        if analysis_results['data_quality']['duplicate_percentage'] > 1:
            f.write("- Remove duplicate rows\n")
        if total_outliers > analysis_results['dataset_info']['total_samples'] * 0.1:
            f.write("- Consider outlier treatment for features with high outlier percentage\n")
        if not analysis_results['class_balance']['is_balanced']:
            f.write("- Consider class balancing techniques\n")
        f.write("- Dataset appears suitable for credit risk analysis\n")
    
    return save_path


def main():
    """
    Main function to run complete dataset analysis.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Ensure directories exist
    PLOTS_DIR.mkdir(exist_ok=True)
    
    try:
        # Analyze dataset
        analysis_results = analyze_credit_dataset()
        
        # Generate plots
        correlation_plot = generate_correlation_matrix_plot(analysis_results['correlation_matrix'])
        class_dist_plot = generate_class_distribution_plot(analysis_results['class_distribution'])
        feature_importance_plot = generate_feature_importance_plot(analysis_results['feature_importance'])
        
        # Save report
        report_path = save_analysis_report(analysis_results)
        
        logger.info("Dataset analysis completed successfully!")
        logger.info(f"Correlation matrix plot: {correlation_plot}")
        logger.info(f"Class distribution plot: {class_dist_plot}")
        logger.info(f"Feature importance plot: {feature_importance_plot}")
        logger.info(f"Analysis report: {report_path}")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error during dataset analysis: {e}")
        raise


if __name__ == "__main__":
    main() 