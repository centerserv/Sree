#!/usr/bin/env python3
"""
Generate Credit Risk Dataset

This script generates a realistic credit risk dataset with the specifications:
- 1000 samples
- 12 features
- 50/50 balanced classes
- 10-15% noise
- 5% outliers
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from data_loader import DataLoader
from data_analysis import CreditDataAnalyzer
from config import setup_logging

def main():
    """Generate and save credit risk dataset."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting credit dataset generation...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader(logger)
    
    # Generate credit dataset
    logger.info("Generating credit risk dataset...")
    X, y = data_loader.create_credit_risk_dataset(n_samples=1000)
    
    # Get feature names
    feature_names = data_loader.credit_feature_names
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save dataset
    dataset_path = data_dir / "credit_risk_dataset.csv"
    df.to_csv(dataset_path, index=False)
    logger.info(f"Dataset saved to {dataset_path}")
    
    # Display dataset info
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Class distribution: {df['target'].value_counts().to_dict()}")
    logger.info(f"Features: {feature_names}")
    
    # Run analysis
    logger.info("Running data analysis...")
    analyzer = CreditDataAnalyzer(logger)
    analysis_results = analyzer.analyze_dataset(X, y, feature_names)
    
    # Generate report
    report = analyzer.generate_report(analysis_results)
    
    # Save report
    report_path = data_dir / "credit_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Analysis report saved to {report_path}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    viz_path = data_dir / "credit_dataset_analysis.png"
    analyzer.create_visualizations(df, str(viz_path))
    logger.info(f"Visualizations saved to {viz_path}")
    
    # Display summary
    print("\n" + "="*60)
    print("CREDIT DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Dataset saved: {dataset_path}")
    print(f"Report saved: {report_path}")
    print(f"Visualizations saved: {viz_path}")
    print("\nDataset Summary:")
    print(f"- Samples: {len(df)}")
    print(f"- Features: {len(feature_names)}")
    print(f"- Good credit: {sum(y == 0)}")
    print(f"- Bad credit: {sum(y == 1)}")
    print(f"- Balance: {sum(y == 0)/len(y)*100:.1f}% / {sum(y == 1)/len(y)*100:.1f}%")
    
    # Show key findings
    print("\nKey Findings:")
    
    # Irrelevant features
    low_corr_features = []
    for feature, corr in analysis_results['correlation_analysis']['target_correlations'].items():
        if abs(corr) < 0.1:
            low_corr_features.append(feature)
    
    if low_corr_features:
        print(f"- Irrelevant features: {', '.join(low_corr_features)}")
    
    # Outliers
    total_outliers = 0
    for data in analysis_results['outlier_analysis']['domain_specific'].values():
        if isinstance(data, dict) and 'indices' in data:
            total_outliers += len(data['indices'])
    
    if total_outliers > 0:
        print(f"- Problematic individuals: {total_outliers}")
    
    # Data quality issues
    issues = analysis_results['data_quality_issues']
    if issues:
        print(f"- Data quality issues: {len(issues)} types found")
    
    print("\n" + "="*60)
    print("Dataset ready for use in SREE dashboard!")
    print("="*60)

if __name__ == "__main__":
    main() 