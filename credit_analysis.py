#!/usr/bin/env python3
"""
SREE Credit Analysis Module
Specialized analysis for banking credit score datasets.
Detects irrelevant columns, problematic individuals, and provides cleaning suggestions.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class CreditAnalyzer:
    """
    Specialized credit analysis using SREE principles.
    Detects data quality issues, irrelevant features, and problematic individuals.
    """
    
    def __init__(self):
        self.irrelevant_patterns = [
            'hair', 'color', 'shoe', 'size', 'height', 'weight', 'eye', 'skin',
            'favorite', 'hobby', 'music', 'movie', 'food', 'sport', 'pet',
            'birthday', 'anniversary', 'zodiac', 'blood_type', 'religion'
        ]
        
        self.critical_credit_features = [
            'income', 'salary', 'revenue', 'earnings', 'wage',
            'credit_score', 'fico', 'credit_history', 'payment_history',
            'debt', 'loan', 'mortgage', 'credit_card', 'bankruptcy',
            'employment', 'job', 'work', 'employer', 'occupation'
        ]
        
        self.suspicious_patterns = [
            'income.*0', 'salary.*0', 'revenue.*0', 'credit_score.*0',
            'payment_history.*none', 'employment.*none', 'job.*none'
        ]
    
    def analyze_dataset(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Comprehensive credit dataset analysis.
        
        Args:
            df: Credit dataset
            target_column: Target variable (e.g., 'default', 'credit_score')
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'dataset_info': self._get_dataset_info(df, target_column),
            'irrelevant_columns': self._detect_irrelevant_columns(df, target_column),
            'problematic_individuals': self._detect_problematic_individuals(df, target_column),
            'data_quality_issues': self._detect_data_quality_issues(df),
            'correlation_analysis': self._analyze_correlations(df, target_column),
            'outlier_analysis': self._detect_outliers(df, target_column),
            'cleaning_suggestions': [],
            'risk_assessment': self._assess_risk(df, target_column)
        }
        
        # Generate cleaning suggestions
        results['cleaning_suggestions'] = self._generate_cleaning_suggestions(results)
        
        return results
    
    def _get_dataset_info(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'target_column': target_column,
            'target_distribution': df[target_column].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
    
    def _detect_irrelevant_columns(self, df: pd.DataFrame, target_column: str) -> List[Dict[str, Any]]:
        """
        Detect columns that are likely irrelevant for credit analysis.
        """
        irrelevant_columns = []
        
        for column in df.columns:
            if column == target_column:
                continue
                
            # Check for irrelevant patterns in column name
            column_lower = column.lower()
            is_irrelevant = any(pattern in column_lower for pattern in self.irrelevant_patterns)
            
            # Check correlation with target
            if df[column].dtype in ['int64', 'float64']:
                correlation = abs(df[column].corr(df[target_column]))
                is_low_correlation = correlation < 0.05
                
                # Check mutual information
                try:
                    mi_score = mutual_info_classif(
                        df[column].values.reshape(-1, 1), 
                        df[target_column], 
                        random_state=42
                    )[0]
                    is_low_mi = mi_score < 0.01
                except:
                    mi_score = 0
                    is_low_mi = True
                
                if is_irrelevant or is_low_correlation or is_low_mi:
                    irrelevant_columns.append({
                        'column': column,
                        'reason': self._get_irrelevance_reason(column, is_irrelevant, correlation, mi_score),
                        'correlation': correlation,
                        'mutual_info': mi_score,
                        'suggestion': 'Consider removing this column'
                    })
        
        return irrelevant_columns
    
    def _get_irrelevance_reason(self, column: str, is_irrelevant: bool, correlation: float, mi_score: float) -> str:
        """Get reason for column irrelevance."""
        reasons = []
        
        if is_irrelevant:
            reasons.append("Column name suggests non-financial data")
        
        if correlation < 0.05:
            reasons.append(f"Very low correlation with target ({correlation:.3f})")
        
        if mi_score < 0.01:
            reasons.append(f"Very low mutual information ({mi_score:.3f})")
        
        return "; ".join(reasons)
    
    def _detect_problematic_individuals(self, df: pd.DataFrame, target_column: str) -> List[Dict[str, Any]]:
        """
        Detect individuals with problematic credit profiles.
        """
        problematic_individuals = []
        
        for idx, row in df.iterrows():
            issues = []
            
            # Check for zero/missing critical values
            for feature in self.critical_credit_features:
                for col in df.columns:
                    if feature in col.lower():
                        if pd.isna(row[col]) or row[col] == 0:
                            issues.append(f"Missing/zero {col}")
            
            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                for col in df.columns:
                    if pattern.split('.*')[0] in col.lower():
                        if pd.isna(row[col]) or row[col] == 0:
                            issues.append(f"Suspicious {col} value")
            
            # Check for extreme outliers
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    z_score = abs((row[col] - df[col].mean()) / df[col].std())
                    if z_score > 4:  # Extreme outlier
                        issues.append(f"Extreme outlier in {col} (z-score: {z_score:.2f})")
            
            if issues:
                problematic_individuals.append({
                    'index': idx,
                    'issues': issues,
                    'risk_level': self._assess_individual_risk(issues),
                    'suggestion': self._get_individual_suggestion(issues)
                })
        
        return problematic_individuals
    
    def _assess_individual_risk(self, issues: List[str]) -> str:
        """Assess risk level for individual."""
        critical_issues = sum(1 for issue in issues if 'income' in issue.lower() or 'credit_score' in issue.lower())
        
        if critical_issues >= 2:
            return "HIGH"
        elif critical_issues >= 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_individual_suggestion(self, issues: List[str]) -> str:
        """Get suggestion for individual."""
        if any('income' in issue.lower() for issue in issues):
            return "Remove individual - missing critical income data"
        elif any('credit_score' in issue.lower() for issue in issues):
            return "Remove individual - missing credit history"
        else:
            return "Review individual data for accuracy"
    
    def _detect_data_quality_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect general data quality issues."""
        issues = {
            'missing_values': {},
            'duplicates': len(df[df.duplicated()]),
            'inconsistent_data_types': [],
            'unusual_distributions': []
        }
        
        # Check missing values
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct > 10:  # More than 10% missing
                issues['missing_values'][col] = missing_pct
        
        # Check for unusual distributions
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check for zero variance
                if df[col].std() == 0:
                    issues['unusual_distributions'].append(f"{col}: Zero variance")
                
                # Check for extreme skewness
                skewness = stats.skew(df[col].dropna())
                if abs(skewness) > 3:
                    issues['unusual_distributions'].append(f"{col}: Highly skewed ({skewness:.2f})")
        
        return issues
    
    def _analyze_correlations(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze correlations with target variable."""
        correlations = {}
        
        for col in df.columns:
            if col != target_column and df[col].dtype in ['int64', 'float64']:
                corr = df[col].corr(df[target_column])
                correlations[col] = {
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'strength': self._get_correlation_strength(abs(corr))
                }
        
        # Sort by absolute correlation
        correlations = dict(sorted(correlations.items(), 
                                 key=lambda x: x[1]['abs_correlation'], 
                                 reverse=True))
        
        return correlations
    
    def _get_correlation_strength(self, abs_corr: float) -> str:
        """Get correlation strength description."""
        if abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.3:
            return "Moderate"
        elif abs_corr >= 0.1:
            return "Weak"
        else:
            return "Very Weak"
    
    def _detect_outliers(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        outliers = {
            'isolation_forest': [],
            'dbscan': [],
            'z_score': [],
            'iqr': []
        }
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_labels = iso_forest.fit_predict(df.select_dtypes(include=[np.number]))
        outliers['isolation_forest'] = np.where(iso_labels == -1)[0].tolist()
        
        # Z-score method
        for col in df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_indices = np.where(z_scores > 3)[0]
            outliers['z_score'].extend(outlier_indices.tolist())
        
        # IQR method
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_indices = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)].index.tolist()
            outliers['iqr'].extend(outlier_indices)
        
        return outliers
    
    def _assess_risk(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Assess overall dataset risk."""
        risk_factors = []
        risk_score = 0
        
        # Check for missing critical features
        critical_features_found = sum(1 for col in df.columns 
                                    for feature in self.critical_credit_features 
                                    if feature in col.lower())
        
        if critical_features_found < 3:
            risk_factors.append("Missing critical credit features")
            risk_score += 30
        
        # Check for high missing values
        total_missing = df.isnull().sum().sum()
        missing_pct = total_missing / (len(df) * len(df.columns)) * 100
        
        if missing_pct > 20:
            risk_factors.append(f"High missing data ({missing_pct:.1f}%)")
            risk_score += 25
        
        # Check for low correlation features
        low_corr_features = sum(1 for col in df.columns 
                              if col != target_column and df[col].dtype in ['int64', 'float64']
                              and abs(df[col].corr(df[target_column])) < 0.05)
        
        if low_corr_features > len(df.columns) * 0.5:
            risk_factors.append("Many low-correlation features")
            risk_score += 20
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': self._get_risk_recommendations(risk_factors)
        }
    
    def _get_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Get recommendations based on risk factors."""
        recommendations = []
        
        for factor in risk_factors:
            if "Missing critical credit features" in factor:
                recommendations.append("Add income, credit score, and payment history columns")
            elif "High missing data" in factor:
                recommendations.append("Implement data imputation or remove incomplete records")
            elif "Many low-correlation features" in factor:
                recommendations.append("Remove irrelevant features to improve model performance")
        
        return recommendations
    
    def _generate_cleaning_suggestions(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate comprehensive cleaning suggestions."""
        suggestions = []
        
        # Suggestions for irrelevant columns
        for col_info in results['irrelevant_columns']:
            suggestions.append({
                'type': 'Remove Column',
                'target': col_info['column'],
                'reason': col_info['reason'],
                'priority': 'Medium'
            })
        
        # Suggestions for problematic individuals
        high_risk_individuals = [ind for ind in results['problematic_individuals'] 
                               if ind['risk_level'] == 'HIGH']
        
        for individual in high_risk_individuals:
            suggestions.append({
                'type': 'Remove Individual',
                'target': f"Row {individual['index']}",
                'reason': f"High risk: {', '.join(individual['issues'])}",
                'priority': 'High'
            })
        
        # Suggestions for data quality issues
        for col, missing_pct in results['data_quality_issues']['missing_values'].items():
            if missing_pct > 50:
                suggestions.append({
                    'type': 'Remove Column',
                    'target': col,
                    'reason': f"Too many missing values ({missing_pct:.1f}%)",
                    'priority': 'High'
                })
            else:
                suggestions.append({
                    'type': 'Impute Values',
                    'target': col,
                    'reason': f"Missing values detected ({missing_pct:.1f}%)",
                    'priority': 'Medium'
                })
        
        return suggestions


def create_credit_analysis_dashboard():
    """Create Streamlit dashboard for credit analysis."""
    st.set_page_config(page_title="SREE Credit Analysis", layout="wide")
    
    st.title("🏦 SREE Credit Analysis Dashboard")
    st.markdown("### Intelligent Credit Dataset Analysis and Cleaning")
    
    # Initialize analyzer
    analyzer = CreditAnalyzer()
    
    # File upload
    st.header("📁 Upload Credit Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with credit data",
        type=['csv'],
        help="Upload your credit dataset (should include features like income, credit_score, etc.)"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column selection
            st.subheader("Column Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                # Auto-detect target column
                target_candidates = [col for col in df.columns 
                                   if col.lower() in ['default', 'credit_score', 'risk', 'target', 'label']]
                default_target = target_candidates[0] if target_candidates else df.columns[-1]
                
                target_column = st.selectbox(
                    "Select target column:",
                    df.columns.tolist(),
                    index=df.columns.tolist().index(default_target),
                    help="Choose the column containing the target variable (default, credit_score, etc.)"
                )
            
            with col2:
                st.info(f"Target column: {target_column}")
                if target_column:
                    unique_values = df[target_column].unique()
                    st.write(f"Unique values: {sorted(unique_values)}")
            
            # Run analysis
            if st.button("🔍 Run Credit Analysis", type="primary"):
                with st.spinner("Analyzing credit dataset..."):
                    results = analyzer.analyze_dataset(df, target_column)
                
                # Display results
                display_credit_analysis_results(results, df)
        
        except Exception as e:
            st.error(f"Error loading file: {e}")


def display_credit_analysis_results(results: Dict[str, Any], df: pd.DataFrame):
    """Display comprehensive credit analysis results."""
    st.header("📊 Credit Analysis Results")
    
    # Dataset Overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", results['dataset_info']['total_samples'])
    with col2:
        st.metric("Total Features", results['dataset_info']['total_features'])
    with col3:
        missing_total = sum(results['dataset_info']['missing_values'].values())
        st.metric("Missing Values", missing_total)
    with col4:
        risk_level = results['risk_assessment']['risk_level']
        st.metric("Risk Level", risk_level, delta_color="inverse")
    
    # Risk Assessment
    st.subheader("⚠️ Risk Assessment")
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.write(f"**Risk Score:** {results['risk_assessment']['risk_score']}/100")
        st.write(f"**Risk Level:** {results['risk_assessment']['risk_level']}")
        
        if results['risk_assessment']['risk_factors']:
            st.write("**Risk Factors:**")
            for factor in results['risk_assessment']['risk_factors']:
                st.write(f"• {factor}")
    
    with risk_col2:
        if results['risk_assessment']['recommendations']:
            st.write("**Recommendations:**")
            for rec in results['risk_assessment']['recommendations']:
                st.write(f"• {rec}")
    
    # Irrelevant Columns
    if results['irrelevant_columns']:
        st.subheader("🚫 Irrelevant Columns Detected")
        
        for col_info in results['irrelevant_columns']:
            with st.expander(f"❌ {col_info['column']}"):
                st.write(f"**Reason:** {col_info['reason']}")
                st.write(f"**Correlation:** {col_info['correlation']:.3f}")
                st.write(f"**Mutual Information:** {col_info['mutual_info']:.3f}")
                st.write(f"**Suggestion:** {col_info['suggestion']}")
    
    # Problematic Individuals
    if results['problematic_individuals']:
        st.subheader("🚨 Problematic Individuals Detected")
        
        # Group by risk level
        high_risk = [ind for ind in results['problematic_individuals'] if ind['risk_level'] == 'HIGH']
        medium_risk = [ind for ind in results['problematic_individuals'] if ind['risk_level'] == 'MEDIUM']
        low_risk = [ind for ind in results['problematic_individuals'] if ind['risk_level'] == 'LOW']
        
        if high_risk:
            st.error(f"🔴 HIGH RISK: {len(high_risk)} individuals")
            for individual in high_risk[:5]:  # Show first 5
                st.write(f"• Row {individual['index']}: {', '.join(individual['issues'])}")
        
        if medium_risk:
            st.warning(f"🟡 MEDIUM RISK: {len(medium_risk)} individuals")
            for individual in medium_risk[:3]:  # Show first 3
                st.write(f"• Row {individual['index']}: {', '.join(individual['issues'])}")
        
        if low_risk:
            st.info(f"🟢 LOW RISK: {len(low_risk)} individuals")
    
    # Cleaning Suggestions
    if results['cleaning_suggestions']:
        st.subheader("🧹 Cleaning Suggestions")
        
        # Group by priority
        high_priority = [s for s in results['cleaning_suggestions'] if s['priority'] == 'High']
        medium_priority = [s for s in results['cleaning_suggestions'] if s['priority'] == 'Medium']
        
        if high_priority:
            st.error("🔴 HIGH PRIORITY")
            for suggestion in high_priority:
                st.write(f"• **{suggestion['type']}:** {suggestion['target']}")
                st.write(f"  Reason: {suggestion['reason']}")
        
        if medium_priority:
            st.warning("🟡 MEDIUM PRIORITY")
            for suggestion in medium_priority:
                st.write(f"• **{suggestion['type']}:** {suggestion['target']}")
                st.write(f"  Reason: {suggestion['reason']}")
    
    # Correlation Analysis
    st.subheader("📈 Feature Correlation Analysis")
    
    if results['correlation_analysis']:
        # Create correlation plot
        corr_data = []
        for col, info in results['correlation_analysis'].items():
            corr_data.append({
                'Feature': col,
                'Correlation': info['correlation'],
                'Strength': info['strength']
            })
        
        corr_df = pd.DataFrame(corr_data)
        
        # Plot top correlations
        fig = px.bar(corr_df.head(10), x='Feature', y='Correlation',
                    color='Strength', title='Top 10 Feature Correlations with Target',
                    color_discrete_map={'Strong': 'green', 'Moderate': 'orange', 
                                      'Weak': 'red', 'Very Weak': 'darkred'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Quality Issues
    if results['data_quality_issues']['missing_values'] or results['data_quality_issues']['unusual_distributions']:
        st.subheader("🔍 Data Quality Issues")
        
        if results['data_quality_issues']['missing_values']:
            st.write("**Missing Values:**")
            for col, pct in results['data_quality_issues']['missing_values'].items():
                st.write(f"• {col}: {pct:.1f}% missing")
        
        if results['data_quality_issues']['unusual_distributions']:
            st.write("**Unusual Distributions:**")
            for issue in results['data_quality_issues']['unusual_distributions']:
                st.write(f"• {issue}")


if __name__ == "__main__":
    create_credit_analysis_dashboard() 