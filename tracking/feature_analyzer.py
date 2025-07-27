#!/usr/bin/env python3
"""
SREE Enhanced Feature Analysis System
Provides deep analysis of each feature including statistical, correlation, and quality metrics.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    """
    Advanced feature analysis system for SREE.
    
    Provides comprehensive analysis of each feature including:
    - Statistical properties and distributions
    - Correlation analysis with target and other features
    - Data quality assessment
    - Feature importance ranking
    - Anomaly detection
    - Feature engineering suggestions
    """
    
    def __init__(self, logs_dir: Path = None):
        """
        Initialize Feature Analyzer.
        
        Args:
            logs_dir: Directory to store analysis logs
        """
        self.logs_dir = logs_dir or Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Analysis results storage
        self.feature_analyses = {}
        self.global_insights = {}
        self.quality_scores = {}
        self.importance_rankings = {}
        
        self.logger = logging.getLogger(__name__)
        
    def analyze_features(self, X: np.ndarray, y: np.ndarray, 
                        feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive feature analysis.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.logger.info(f"Starting comprehensive feature analysis for {len(feature_names)} features")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Perform individual feature analysis
        for i, feature_name in enumerate(feature_names):
            self.feature_analyses[feature_name] = self._analyze_single_feature(
                df, feature_name, i
            )
        
        # Perform global analysis
        self.global_insights = self._perform_global_analysis(df, feature_names)
        
        # Calculate quality scores
        self.quality_scores = self._calculate_quality_scores(feature_names)
        
        # Calculate importance rankings
        self.importance_rankings = self._calculate_importance_rankings(df, feature_names)
        
        return {
            "feature_analyses": self.feature_analyses,
            "global_insights": self.global_insights,
            "quality_scores": self.quality_scores,
            "importance_rankings": self.importance_rankings
        }
    
    def _analyze_single_feature(self, df: pd.DataFrame, feature_name: str, 
                               feature_idx: int) -> Dict[str, Any]:
        """Analyze a single feature comprehensively."""
        feature_data = df[feature_name]
        target_data = df['target']
        
        analysis = {
            "feature_name": feature_name,
            "feature_index": feature_idx,
            "data_type": str(feature_data.dtype),
            "basic_stats": self._calculate_basic_stats(feature_data),
            "distribution_analysis": self._analyze_distribution(feature_data),
            "correlation_analysis": self._analyze_correlations(feature_data, target_data),
            "quality_analysis": self._analyze_quality(feature_data),
            "anomaly_analysis": self._detect_anomalies(feature_data),
            "target_relationship": self._analyze_target_relationship(feature_data, target_data),
            "feature_engineering": self._suggest_feature_engineering(feature_data, feature_name)
        }
        
        return analysis
    
    def _calculate_basic_stats(self, feature_data: pd.Series) -> Dict[str, Any]:
        """Calculate basic statistical properties."""
        stats_dict = {
            "count": int(feature_data.count()),
            "missing_count": int(feature_data.isnull().sum()),
            "missing_percentage": float(feature_data.isnull().sum() / len(feature_data) * 100),
            "unique_values": int(feature_data.nunique()),
            "cardinality": float(feature_data.nunique() / len(feature_data)),
            "mean": float(feature_data.mean()) if feature_data.dtype in ['float64', 'int64'] else None,
            "median": float(feature_data.median()) if feature_data.dtype in ['float64', 'int64'] else None,
            "std": float(feature_data.std()) if feature_data.dtype in ['float64', 'int64'] else None,
            "min": float(feature_data.min()) if feature_data.dtype in ['float64', 'int64'] else None,
            "max": float(feature_data.max()) if feature_data.dtype in ['float64', 'int64'] else None,
            "q25": float(feature_data.quantile(0.25)) if feature_data.dtype in ['float64', 'int64'] else None,
            "q75": float(feature_data.quantile(0.75)) if feature_data.dtype in ['float64', 'int64'] else None,
            "skewness": float(stats.skew(feature_data.dropna())) if feature_data.dtype in ['float64', 'int64'] else None,
            "kurtosis": float(stats.kurtosis(feature_data.dropna())) if feature_data.dtype in ['float64', 'int64'] else None
        }
        
        return stats_dict
    
    def _analyze_distribution(self, feature_data: pd.Series) -> Dict[str, Any]:
        """Analyze feature distribution."""
        if feature_data.dtype not in ['float64', 'int64']:
            return {"distribution_type": "categorical", "analysis": "Not applicable for categorical data"}
        
        # Test for normal distribution
        try:
            _, p_value = stats.normaltest(feature_data.dropna())
            is_normal = p_value > 0.05
        except:
            is_normal = False
            p_value = 0.0
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {f"p{p}": float(feature_data.quantile(p/100)) for p in percentiles}
        
        # Detect distribution type
        skewness = abs(stats.skew(feature_data.dropna()))
        kurtosis = abs(stats.kurtosis(feature_data.dropna()))
        
        if is_normal:
            distribution_type = "normal"
        elif skewness > 1:
            distribution_type = "skewed"
        elif kurtosis > 3:
            distribution_type = "heavy_tailed"
        else:
            distribution_type = "other"
        
        return {
            "distribution_type": distribution_type,
            "is_normal": is_normal,
            "normality_p_value": float(p_value),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "percentiles": percentile_values,
            "iqr": float(feature_data.quantile(0.75) - feature_data.quantile(0.25)),
            "range": float(feature_data.max() - feature_data.min())
        }
    
    def _analyze_correlations(self, feature_data: pd.Series, target_data: pd.Series) -> Dict[str, Any]:
        """Analyze correlations with target and other features."""
        correlations = {}
        
        # Correlation with target
        if feature_data.dtype in ['float64', 'int64'] and target_data.dtype in ['float64', 'int64']:
            try:
                pearson_corr = float(feature_data.corr(target_data))
                spearman_corr = float(feature_data.corr(target_data, method='spearman'))
                kendall_corr = float(feature_data.corr(target_data, method='kendall'))
                
                correlations["target_correlation"] = {
                    "pearson": pearson_corr,
                    "spearman": spearman_corr,
                    "kendall": kendall_corr,
                    "abs_pearson": abs(pearson_corr),
                    "correlation_strength": self._get_correlation_strength(abs(pearson_corr))
                }
            except:
                correlations["target_correlation"] = {"error": "Could not calculate correlation"}
        
        # Mutual information with target
        try:
            mi_score = mutual_info_classif(feature_data.values.reshape(-1, 1), target_data)[0]
            correlations["mutual_information"] = float(mi_score)
        except:
            correlations["mutual_information"] = None
        
        return correlations
    
    def _analyze_quality(self, feature_data: pd.Series) -> Dict[str, Any]:
        """Analyze data quality issues."""
        quality_issues = []
        quality_score = 100.0
        
        # Check for missing values
        missing_pct = feature_data.isnull().sum() / len(feature_data) * 100
        if missing_pct > 0:
            quality_issues.append(f"Missing values: {missing_pct:.2f}%")
            quality_score -= missing_pct * 2  # Heavy penalty for missing values
        
        # Check for duplicates
        duplicate_pct = (len(feature_data) - feature_data.nunique()) / len(feature_data) * 100
        if duplicate_pct > 50:
            quality_issues.append(f"High duplicate rate: {duplicate_pct:.2f}%")
            quality_score -= duplicate_pct * 0.5
        
        # Check for outliers (if numerical)
        if feature_data.dtype in ['float64', 'int64']:
            outliers = self._detect_outliers_iqr(feature_data)
            outlier_pct = len(outliers) / len(feature_data) * 100
            if outlier_pct > 5:
                quality_issues.append(f"High outlier rate: {outlier_pct:.2f}%")
                quality_score -= outlier_pct * 0.3
        
        # Check for constant values
        if feature_data.nunique() == 1:
            quality_issues.append("Constant feature (no variance)")
            quality_score = 0.0
        
        # Check for low variance
        if feature_data.dtype in ['float64', 'int64']:
            cv = feature_data.std() / feature_data.mean() if feature_data.mean() != 0 else 0
            if cv < 0.01:
                quality_issues.append("Very low coefficient of variation")
                quality_score -= 20.0
        
        return {
            "quality_score": max(0.0, quality_score),
            "quality_issues": quality_issues,
            "missing_percentage": float(missing_pct),
            "duplicate_percentage": float(duplicate_pct),
            "outlier_percentage": float(outlier_pct) if feature_data.dtype in ['float64', 'int64'] else 0.0,
            "is_constant": feature_data.nunique() == 1,
            "coefficient_of_variation": float(cv) if feature_data.dtype in ['float64', 'int64'] else None
        }
    
    def _detect_anomalies(self, feature_data: pd.Series) -> Dict[str, Any]:
        """Detect anomalies in the feature."""
        anomalies = {}
        
        if feature_data.dtype not in ['float64', 'int64']:
            return {"anomaly_type": "categorical", "analysis": "Not applicable for categorical data"}
        
        # IQR method
        iqr_outliers = self._detect_outliers_iqr(feature_data)
        anomalies["iqr_outliers"] = {
            "count": len(iqr_outliers),
            "percentage": len(iqr_outliers) / len(feature_data) * 100,
            "indices": iqr_outliers.index.tolist()
        }
        
        # Z-score method
        z_outliers = self._detect_outliers_zscore(feature_data)
        anomalies["zscore_outliers"] = {
            "count": len(z_outliers),
            "percentage": len(z_outliers) / len(feature_data) * 100,
            "indices": z_outliers.index.tolist()
        }
        
        # Isolation Forest
        try:
            iso_outliers = self._detect_outliers_isolation_forest(feature_data)
            anomalies["isolation_forest_outliers"] = {
                "count": len(iso_outliers),
                "percentage": len(iso_outliers) / len(feature_data) * 100,
                "indices": iso_outliers.index.tolist()
            }
        except:
            anomalies["isolation_forest_outliers"] = {"error": "Could not perform isolation forest analysis"}
        
        return anomalies
    
    def _detect_outliers_iqr(self, feature_data: pd.Series, threshold: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = feature_data.quantile(0.25)
        Q3 = feature_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
    
    def _detect_outliers_zscore(self, feature_data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(feature_data.dropna()))
        outlier_indices = np.where(z_scores > threshold)[0]
        return feature_data.iloc[outlier_indices]
    
    def _detect_outliers_isolation_forest(self, feature_data: pd.Series) -> pd.Series:
        """Detect outliers using Isolation Forest."""
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(feature_data.values.reshape(-1, 1))
        outlier_indices = np.where(predictions == -1)[0]
        return feature_data.iloc[outlier_indices]
    
    def _analyze_target_relationship(self, feature_data: pd.Series, target_data: pd.Series) -> Dict[str, Any]:
        """Analyze relationship with target variable."""
        relationship = {}
        
        if feature_data.dtype in ['float64', 'int64'] and target_data.dtype in ['float64', 'int64']:
            # Linear relationship
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(feature_data.dropna(), target_data[feature_data.dropna().index])
                relationship["linear_relationship"] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "std_error": float(std_err)
                }
            except:
                relationship["linear_relationship"] = {"error": "Could not calculate linear relationship"}
        
        # Target distribution by feature (for categorical features)
        if feature_data.dtype == 'object' or feature_data.nunique() < 10:
            target_by_feature = target_data.groupby(feature_data).agg(['mean', 'count', 'std'])
            relationship["target_distribution"] = target_by_feature.to_dict()
        
        return relationship
    
    def _suggest_feature_engineering(self, feature_data: pd.Series, feature_name: str) -> Dict[str, Any]:
        """Suggest feature engineering techniques."""
        suggestions = []
        
        if feature_data.dtype in ['float64', 'int64']:
            # Check for skewness
            skewness = abs(stats.skew(feature_data.dropna()))
            if skewness > 1:
                suggestions.append("Consider log transformation or Box-Cox transformation")
            
            # Check for outliers
            outlier_pct = len(self._detect_outliers_iqr(feature_data)) / len(feature_data) * 100
            if outlier_pct > 5:
                suggestions.append("Consider outlier handling (capping, winsorization)")
            
            # Check for scaling needs
            if feature_data.std() > feature_data.mean() * 10:
                suggestions.append("Consider standardization or normalization")
            
            # Check for binning opportunities
            if feature_data.nunique() > 20:
                suggestions.append("Consider binning for better interpretability")
        
        # Check for missing values
        if feature_data.isnull().sum() > 0:
            suggestions.append("Implement missing value imputation strategy")
        
        # Check for encoding needs (categorical)
        if feature_data.dtype == 'object':
            suggestions.append("Apply categorical encoding (one-hot, label, target)")
        
        return {
            "suggestions": suggestions,
            "priority": "high" if len(suggestions) > 3 else "medium" if len(suggestions) > 1 else "low"
        }
    
    def _perform_global_analysis(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Perform global analysis across all features."""
        global_analysis = {
            "dataset_overview": {
                "total_features": len(feature_names),
                "total_samples": len(df),
                "target_distribution": df['target'].value_counts().to_dict(),
                "missing_data_overview": df.isnull().sum().to_dict()
            },
            "feature_correlations": self._calculate_feature_correlations(df, feature_names),
            "multicollinearity": self._detect_multicollinearity(df, feature_names),
            "feature_redundancy": self._detect_feature_redundancy(df, feature_names)
        }
        
        return global_analysis
    
    def _calculate_feature_correlations(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate correlation matrix and identify highly correlated features."""
        numerical_features = df[feature_names].select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numerical_features].corr()
        
        # Find highly correlated feature pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_correlations.append({
                        "feature1": correlation_matrix.columns[i],
                        "feature2": correlation_matrix.columns[j],
                        "correlation": float(corr_value)
                    })
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlations": high_correlations,
            "avg_correlation": float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean())
        }
    
    def _detect_multicollinearity(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Detect multicollinearity using VIF."""
        numerical_features = df[feature_names].select_dtypes(include=[np.number]).columns
        
        if len(numerical_features) < 2:
            return {"vif_scores": {}, "multicollinearity_issues": []}
        
        # Calculate VIF for each feature
        vif_scores = {}
        for feature in numerical_features:
            try:
                other_features = [f for f in numerical_features if f != feature]
                if len(other_features) > 0:
                    X = df[other_features]
                    y = df[feature]
                    
                    # Simple linear regression for VIF calculation
                    X_with_const = np.column_stack([np.ones(len(X)), X])
                    beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
                    y_pred = X_with_const @ beta
                    residuals = y - y_pred
                    mse = np.sum(residuals**2) / (len(y) - len(beta))
                    vif = 1 / (1 - (np.corrcoef(y, y_pred)[0, 1]**2))
                    vif_scores[feature] = float(vif)
                else:
                    vif_scores[feature] = 1.0
            except:
                vif_scores[feature] = None
        
        # Identify problematic features
        multicollinearity_issues = [feature for feature, vif in vif_scores.items() 
                                  if vif is not None and vif > 10]
        
        return {
            "vif_scores": vif_scores,
            "multicollinearity_issues": multicollinearity_issues,
            "avg_vif": float(np.mean([v for v in vif_scores.values() if v is not None]))
        }
    
    def _detect_feature_redundancy(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Detect redundant features."""
        redundant_features = []
        
        for i, feature1 in enumerate(feature_names):
            for j, feature2 in enumerate(feature_names[i+1:], i+1):
                # Check if features are identical
                if df[feature1].equals(df[feature2]):
                    redundant_features.append({
                        "feature1": feature1,
                        "feature2": feature2,
                        "type": "identical",
                        "similarity": 1.0
                    })
                # Check for high similarity
                elif df[feature1].dtype in ['float64', 'int64'] and df[feature2].dtype in ['float64', 'int64']:
                    correlation = abs(df[feature1].corr(df[feature2]))
                    if correlation > 0.95:
                        redundant_features.append({
                            "feature1": feature1,
                            "feature2": feature2,
                            "type": "highly_correlated",
                            "similarity": float(correlation)
                        })
        
        return {
            "redundant_features": redundant_features,
            "total_redundant_pairs": len(redundant_features)
        }
    
    def _calculate_quality_scores(self, feature_names: List[str]) -> Dict[str, float]:
        """Calculate overall quality scores for each feature."""
        quality_scores = {}
        
        for feature_name in feature_names:
            if feature_name in self.feature_analyses:
                analysis = self.feature_analyses[feature_name]
                quality_analysis = analysis["quality_analysis"]
                quality_scores[feature_name] = quality_analysis["quality_score"]
        
        return quality_scores
    
    def _calculate_importance_rankings(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate feature importance rankings using multiple methods."""
        numerical_features = df[feature_names].select_dtypes(include=[np.number]).columns
        
        if len(numerical_features) == 0:
            return {"error": "No numerical features for importance calculation"}
        
        importance_scores = {}
        
        # Correlation-based importance
        correlations = []
        for feature in numerical_features:
            corr = abs(df[feature].corr(df['target']))
            correlations.append((feature, corr))
        
        correlation_ranking = sorted(correlations, key=lambda x: x[1], reverse=True)
        
        # Mutual information importance
        try:
            mi_scores = mutual_info_classif(df[numerical_features], df['target'])
            mi_ranking = [(feature, score) for feature, score in zip(numerical_features, mi_scores)]
            mi_ranking = sorted(mi_ranking, key=lambda x: x[1], reverse=True)
        except:
            mi_ranking = []
        
        # F-statistic importance
        try:
            f_scores, _ = f_classif(df[numerical_features], df['target'])
            f_ranking = [(feature, score) for feature, score in zip(numerical_features, f_scores)]
            f_ranking = sorted(f_ranking, key=lambda x: x[1], reverse=True)
        except:
            f_ranking = []
        
        return {
            "correlation_ranking": correlation_ranking,
            "mutual_information_ranking": mi_ranking,
            "f_statistic_ranking": f_ranking,
            "top_features": {
                "by_correlation": [feature for feature, _ in correlation_ranking[:5]],
                "by_mutual_information": [feature for feature, _ in mi_ranking[:5]] if mi_ranking else [],
                "by_f_statistic": [feature for feature, _ in f_ranking[:5]] if f_ranking else []
            }
        }
    
    def _get_correlation_strength(self, correlation: float) -> str:
        """Get correlation strength description."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def generate_feature_visualization(self, save_path: str = None) -> str:
        """Generate comprehensive feature analysis visualization."""
        if not self.feature_analyses:
            return ""
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Quality scores
        ax1 = axes[0, 0]
        features = list(self.quality_scores.keys())
        scores = list(self.quality_scores.values())
        
        colors = ['green' if score > 80 else 'orange' if score > 60 else 'red' for score in scores]
        bars = ax1.bar(features, scores, color=colors, alpha=0.7)
        
        ax1.set_title('Feature Quality Scores', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Quality Score (0-100)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Importance rankings
        ax2 = axes[0, 1]
        if self.importance_rankings and "correlation_ranking" in self.importance_rankings:
            top_features = self.importance_rankings["correlation_ranking"][:10]
            feature_names = [f[0] for f in top_features]
            importance_scores = [f[1] for f in top_features]
            
            bars = ax2.barh(feature_names, importance_scores, color='skyblue', alpha=0.7)
            ax2.set_title('Top 10 Features by Correlation', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Absolute Correlation')
            
            # Add value labels
            for bar, score in zip(bars, importance_scores):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        # Distribution analysis
        ax3 = axes[1, 0]
        if self.global_insights and "feature_correlations" in self.global_insights:
            corr_matrix = pd.DataFrame(self.global_insights["feature_correlations"]["correlation_matrix"])
            if not corr_matrix.empty:
                im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
                ax3.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
                ax3.set_xticks(range(len(corr_matrix.columns)))
                ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                ax3.set_yticks(range(len(corr_matrix.index)))
                ax3.set_yticklabels(corr_matrix.index)
                plt.colorbar(im, ax=ax3, label='Correlation')
        
        # Missing data overview
        ax4 = axes[1, 1]
        if self.global_insights and "dataset_overview" in self.global_insights:
            missing_data = self.global_insights["dataset_overview"]["missing_data_overview"]
            features = list(missing_data.keys())
            missing_counts = list(missing_data.values())
            
            bars = ax4.bar(features, missing_counts, color='lightcoral', alpha=0.7)
            ax4.set_title('Missing Data by Feature', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Features')
            ax4.set_ylabel('Missing Count')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.logs_dir / f"feature_analysis_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def save_analysis_logs(self, filename: str = None) -> str:
        """Save feature analysis logs to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feature_analysis_logs_{timestamp}.json"
        
        filepath = self.logs_dir / filename
        
        logs_data = {
            "feature_analyses": self.feature_analyses,
            "global_insights": self.global_insights,
            "quality_scores": self.quality_scores,
            "importance_rankings": self.importance_rankings,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_features_analyzed": len(self.feature_analyses)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(logs_data, f, indent=2, default=str)
        
        return str(filepath)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get a summary report of feature analysis."""
        if not self.feature_analyses:
            return {"error": "No feature analysis performed"}
        
        # Calculate overall statistics
        quality_scores = list(self.quality_scores.values())
        avg_quality = np.mean(quality_scores)
        
        # Identify problematic features
        problematic_features = [feature for feature, score in self.quality_scores.items() if score < 60]
        
        # Top features by importance
        top_features = []
        if self.importance_rankings and "correlation_ranking" in self.importance_rankings:
            top_features = [feature for feature, _ in self.importance_rankings["correlation_ranking"][:5]]
        
        # Data quality issues
        quality_issues = []
        for feature, analysis in self.feature_analyses.items():
            quality_analysis = analysis["quality_analysis"]
            if quality_analysis["quality_score"] < 80:
                quality_issues.extend(quality_analysis["quality_issues"])
        
        return {
            "total_features": len(self.feature_analyses),
            "average_quality_score": float(avg_quality),
            "problematic_features": problematic_features,
            "top_features": top_features,
            "quality_issues": list(set(quality_issues)),  # Remove duplicates
            "analysis_timestamp": datetime.now().isoformat()
        } 