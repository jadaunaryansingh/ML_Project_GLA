import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'bixuchenggong11.CSV')
# Page configuration
st.set_page_config(
    page_title="Sertraline-like Activity Predictor",
    page_icon="💊",
    layout="wide"
)
# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">🧬 Machine Learning Model for Predicting Sertraline-like Activities</p>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
Predict sertraline-like antidepressant activity and cancer chemosensitization potential using multiple ML algorithms
</div>
""", unsafe_allow_html=True)

# Function to calculate molecular descriptors
def calculate_descriptors(smiles, raise_errors=False):
    """Calculate comprehensive molecular descriptors from SMILES string"""
    try:
        if pd.isna(smiles):
            return None
        
        # Convert to string and clean SMILES - remove ALL whitespace
        smiles_str = str(smiles).strip()
        if smiles_str == '' or smiles_str == 'nan':
            return None
        
        # Remove all whitespace characters (spaces, tabs, newlines, etc.)
        smiles_clean = ''.join(smiles_str.split())
        
        if len(smiles_clean) == 0:
            return None
            
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            return None
        
        # Calculate all descriptors
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'MolMR': Descriptors.MolMR(mol),
            'BalabanJ': Descriptors.BalabanJ(mol),
            'BertzCT': Descriptors.BertzCT(mol),
            'Chi0': Descriptors.Chi0(mol),
            'Chi1': Descriptors.Chi1(mol),
            'HallKierAlpha': Descriptors.HallKierAlpha(mol),
            'Kappa1': Descriptors.Kappa1(mol),
            'Kappa2': Descriptors.Kappa2(mol),
            'Kappa3': Descriptors.Kappa3(mol),
            'LabuteASA': Descriptors.LabuteASA(mol),
            'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
            'PEOE_VSA2': Descriptors.PEOE_VSA2(mol),
            'SMR_VSA1': Descriptors.SMR_VSA1(mol),
            'SMR_VSA2': Descriptors.SMR_VSA2(mol),
            'SlogP_VSA1': Descriptors.SlogP_VSA1(mol),
            'SlogP_VSA2': Descriptors.SlogP_VSA2(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles(mol),
            'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles(mol),
            'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
            'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
        }
        return descriptors
    except Exception as e:
        if raise_errors:
            raise
        # Return None on any error (silent failure)
        return None

@st.cache_data
def load_and_prepare_data(file_path):
    """Load dataset and prepare features"""
    try:
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_columns = ['activity', 'smiles']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info(f"Available columns: {', '.join(df.columns.tolist())}")
            return None
        
        # Clean data
        initial_count = len(df)
        df = df.dropna(subset=['activity', 'smiles'])
        # Clean SMILES strings - remove all whitespace
        df['smiles'] = df['smiles'].astype(str).str.strip()
        df = df[df['smiles'] != '']
        df = df[df['smiles'] != 'nan']
        
        if len(df) == 0:
            st.error("❌ No valid rows found after cleaning the dataset")
            return None
        
        # Convert activity to int
        try:
            df['activity'] = df['activity'].astype(int)
        except (ValueError, TypeError):
            st.error("❌ Could not convert 'activity' column to integers")
            return None
        
        # Check for valid activity values
        if not df['activity'].isin([0, 1]).all():
            st.warning("⚠️ Activity column contains values other than 0 and 1. Filtering to 0/1 only.")
            df = df[df['activity'].isin([0, 1])]
        
        if len(df) == 0:
            st.error("❌ No valid rows with activity 0 or 1")
            return None
        
        if initial_count > len(df):
            st.info(f"ℹ️ Filtered dataset from {initial_count} to {len(df)} valid rows")
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def prepare_features(df, show_progress=True):
    """Extract features from SMILES and prepare dataset"""
    features_list = []
    valid_indices = []
    error_count = 0
    failed_smiles = []
    
    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for idx, row in df.iterrows():
        if show_progress:
            # Handle missing 'name' column gracefully
            compound_name = row.get('name', f'Compound {idx+1}') if 'name' in row else f'Compound {idx+1}'
            status_text.text(f"Processing compound {idx+1}/{len(df)}: {compound_name}")
            progress_bar.progress((idx + 1) / len(df))
        
        try:
            # Clean the SMILES string before processing
            smiles_raw = str(row['smiles']).strip()
            smiles_clean = ''.join(smiles_raw.split())  # Remove all whitespace
            
            if len(smiles_clean) == 0:
                error_count += 1
                continue
            
            descriptors = calculate_descriptors(smiles_clean)
            if descriptors is not None:
                features_list.append(descriptors)
                valid_indices.append(idx)
            else:
                error_count += 1
                if len(failed_smiles) < 5:  # Keep track of first few failures
                    failed_smiles.append(smiles_clean)
        except Exception as e:
            # Skip compounds that cause errors
            error_count += 1
            if len(failed_smiles) < 5:
                failed_smiles.append(f"Error: {str(e)}")
            continue
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    # Create DataFrame from features_list
    if len(features_list) == 0:
        # Return empty DataFrame with no columns if no features were extracted
        features_df = pd.DataFrame()
        df_filtered = pd.DataFrame()
        
        # Store error info in session state for debugging
        if show_progress:
            st.session_state['feature_extraction_errors'] = {
                'total_processed': len(df),
                'errors': error_count,
                'failed_smiles': failed_smiles
            }
    else:
        features_df = pd.DataFrame(features_list)
        if len(valid_indices) > 0:
            df_filtered = df.iloc[valid_indices].reset_index(drop=True)
        else:
            df_filtered = pd.DataFrame()
    
    return features_df, df_filtered

def train_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results"""
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=5,
            gamma='scale',
            probability=True,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
    }
    
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        progress_bar.progress((idx + 1) / len(models))
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Metrics
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def optimize_svm_with_bagging(X_train, X_test, y_train, y_test):
    """Optimize SVM using Bagging ensemble method (17 estimators as per paper)"""
    
    base_svm = SVC(kernel='rbf', C=5, probability=True, random_state=42)
    bagging_svm = BaggingClassifier(
        estimator=base_svm,
        n_estimators=17,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    with st.spinner("Training Bagging SVM (17 estimators)..."):
        bagging_svm.fit(X_train, y_train)
    
    y_pred = bagging_svm.predict(X_test)
    y_pred_proba = bagging_svm.predict_proba(X_test)[:, 1]
    
    cv_scores = cross_val_score(bagging_svm, X_train, y_train, cv=5, scoring='accuracy')
    
    return {
        'model': bagging_svm,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Inactive (0)', 'Active (1)'],
                yticklabels=['Inactive (0)', 'Active (1)'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def plot_roc_curves(results, bagging_result=None):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC = {roc_auc:.3f})', 
                color=colors[idx])
    
    if bagging_result:
        fpr, tpr, _ = roc_curve(bagging_result['y_test'], bagging_result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=3, label=f'Bagging SVM (AUC = {roc_auc:.3f})', 
                color='#e377c2', linestyle='--')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves - Model Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(11, 7))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        bars = ax.barh(range(len(importance_df)), importance_df['Importance'], color=colors)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'], fontsize=11)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Important Features - {model_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_df['Importance'])):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    return None

def plot_comparison_bar_chart(results_df):
    """Plot model comparison bar chart"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
        values = [float(v.strip('%')) for v in results_df[metric]]
        bars = ax.bar(results_df['Model'], values, color=color, alpha=0.7, edgecolor='black')
        ax.set_ylabel(f'{metric} (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Model Comparison - {metric}', fontsize=13, fontweight='bold', pad=10)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main Application
def main():
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info("""
    This model uses:
    - **31 molecular descriptors** from RDKit
    - **5 ML algorithms** + Bagging SVM
    - **268 compounds** from research data
    - **Cross-validation** for robust evaluation
    """)
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        df = load_and_prepare_data(DATASET_PATH)
    
    if df is None:
        st.error(f"❌ Could not load dataset from: {DATASET_PATH}")
        st.info("Please update the DATASET_PATH variable at the top of the code.")
        return
    
    # Dataset Overview
    st.header("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    active_count = int(df['activity'].sum())
    inactive_count = len(df) - active_count
    activity_ratio = (active_count / len(df)) * 100
    
    with col1:
        st.metric("Total Compounds", len(df), help="Total number of compounds in dataset")
    with col2:
        st.metric("Active Compounds", active_count, help="Compounds with activity = 1")
    with col3:
        st.metric("Inactive Compounds", inactive_count, help="Compounds with activity = 0")
    with col4:
        st.metric("Activity Ratio", f"{activity_ratio:.1f}%", help="Percentage of active compounds")
    
    # Show sample data
    with st.expander("📊 View Sample Dataset", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Train Models Button
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        
        # Feature Extraction
        st.header("🔬 Step 1: Feature Extraction")
        with st.spinner("Extracting molecular descriptors from SMILES..."):
            features_df, df_filtered = prepare_features(df, show_progress=True)
        
        # Check if features were successfully extracted
        if features_df is None:
            st.error("❌ Feature extraction returned None. Please check your dataset.")
            return
        
        # Check if DataFrame has columns (more robust check)
        if features_df.empty or len(features_df.columns) == 0 or features_df.shape[1] == 0:
            st.error("❌ No valid features could be extracted from the SMILES strings.")
            
            # Show detailed error information
            error_info = st.session_state.get('feature_extraction_errors', {})
            if error_info:
                st.warning(f"Processed {error_info.get('total_processed', 0)} compounds, {error_info.get('errors', 0)} failed")
                if error_info.get('failed_smiles'):
                    st.write("**Sample failed SMILES:**")
                    for failed in error_info['failed_smiles'][:3]:
                        st.code(failed)
            
            st.info("💡 **Possible causes:**")
            st.info("1. SMILES strings in your dataset may be invalid or malformed")
            st.info("2. RDKit may not be able to parse the SMILES format")
            st.info("3. Check that the 'smiles' column contains valid SMILES notation")
            
            # Test a sample SMILES directly to verify RDKit works
            with st.expander("🔍 Debug: Test SMILES Parsing"):
                st.write("**Testing with first SMILES from dataset:**")
                if len(df) > 0 and 'smiles' in df.columns:
                    test_smiles_raw = str(df['smiles'].iloc[0]).strip()
                    test_smiles_clean = ''.join(test_smiles_raw.split())
                    st.code(f"Original: {test_smiles_raw}")
                    st.code(f"Cleaned: {test_smiles_clean}")
                    
                    # Test parsing
                    test_mol = Chem.MolFromSmiles(test_smiles_clean)
                    if test_mol is None:
                        st.error(f"❌ RDKit cannot parse: {test_smiles_clean}")
                    else:
                        st.success(f"✅ RDKit can parse this SMILES")
                        
                        # Try calculating descriptors
                        try:
                            test_desc = calculate_descriptors(test_smiles_clean, raise_errors=True)
                            if test_desc is None:
                                st.error("❌ calculate_descriptors returned None even though molecule parsed successfully")
                            else:
                                st.success(f"✅ Descriptors calculated: {len(test_desc)} features")
                                st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in list(test_desc.items())[:5]})
                        except Exception as e:
                            st.error(f"❌ Error calculating descriptors: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            st.write("**This is the actual error preventing feature extraction!**")
                
                st.markdown("---")
                st.write("**Testing RDKit installation with simple SMILES:**")
                simple_test = "CCO"  # Ethanol
                simple_mol = Chem.MolFromSmiles(simple_test)
                if simple_mol is None:
                    st.error("❌ RDKit cannot parse even simple SMILES. RDKit installation may be corrupted.")
                else:
                    st.success(f"✅ RDKit is working (tested with: {simple_test})")
            return
        
        if len(df_filtered) == 0 or df_filtered.empty:
            st.error("❌ No valid compounds found after feature extraction. Please check your dataset.")
            return
        
        # Check if we have enough samples
        if len(df_filtered) < 10:
            st.warning(f"⚠️ Only {len(df_filtered)} valid compounds found. This may not be enough for reliable model training.")
        
        st.success(f"✅ Successfully extracted {features_df.shape[1]} features from {len(df_filtered)} valid compounds")
        
        with st.expander("📊 Feature Statistics", expanded=False):
            # Double check before calling describe
            if len(features_df.columns) > 0 and features_df.shape[0] > 0:
                try:
                    st.dataframe(features_df.describe(), use_container_width=True)
                except ValueError as e:
                    st.warning(f"Could not generate statistics: {str(e)}")
            else:
                st.warning("No features available to display statistics.")
        
        # Prepare Data
        st.header("🔧 Step 2: Data Preparation")
        X = features_df.values
        y = df_filtered['activity'].values
        
        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-Test Split
        # Check if stratification is possible (need at least 2 samples per class)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        can_stratify = len(unique_classes) > 1 and all(count >= 2 for count in class_counts)
        
        if can_stratify:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
                )
            except ValueError:
                # Fallback if stratification fails
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=random_state
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        with col3:
            st.metric("Features", X_train.shape[1])
        
        # Train Models
        st.header("🤖 Step 3: Model Training & Evaluation")
        results = train_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Display Results Table
        st.subheader("📈 Model Performance Metrics")
        results_data = []
        for name, result in results.items():
            results_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']*100:.2f}%",
                'Precision': f"{result['precision']*100:.2f}%",
                'Recall': f"{result['recall']*100:.2f}%",
                'F1-Score': f"{result['f1_score']*100:.2f}%",
                'CV Mean': f"{result['cv_mean']*100:.2f}%",
                'CV Std': f"±{result['cv_std']*100:.2f}%"
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Highlight best model
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_accuracy = results[best_model_name]['accuracy']
        
        st.dataframe(results_df, use_container_width=True)
        st.success(f"🏆 **Best Model: {best_model_name}** with **{best_accuracy*100:.2f}%** accuracy")
        
        # Model Comparison Chart
        st.subheader("📊 Model Comparison Visualization")
        fig_comparison = plot_comparison_bar_chart(results_df)
        st.pyplot(fig_comparison)
        plt.close()
        
        # SVM Optimization with Bagging
        st.header("🎯 Step 4: SVM Optimization with Bagging")
        st.markdown("*Training ensemble of 17 SVM models as described in the research paper...*")
        
        bagging_result = optimize_svm_with_bagging(X_train, X_test, y_train, y_test)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{bagging_result['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{bagging_result['precision']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{bagging_result['recall']*100:.2f}%")
        with col4:
            st.metric("F1-Score", f"{bagging_result['f1_score']*100:.2f}%")
        with col5:
            st.metric("CV Score", f"{bagging_result['cv_mean']*100:.2f}%")
        
        # Compare with base SVM
        improvement = (bagging_result['accuracy'] - results['SVM']['accuracy']) * 100
        if improvement > 0:
            st.success(f"✨ Bagging improved SVM accuracy by **{improvement:.2f}%**")
        else:
            st.info(f"ℹ️ Bagging accuracy: {bagging_result['accuracy']*100:.2f}% vs Base SVM: {results['SVM']['accuracy']*100:.2f}%")
        
        # ROC Curves
        st.header("📊 Step 5: ROC Curve Analysis")
        fig_roc = plot_roc_curves(results, bagging_result)
        st.pyplot(fig_roc)
        plt.close()
        
        # Confusion Matrices
        st.header("🔍 Step 6: Confusion Matrix Analysis")
        
        col1, col2, col3 = st.columns(3)
        model_names = list(results.keys())
        
        for idx, name in enumerate(model_names):
            with [col1, col2, col3][idx % 3]:
                fig_cm = plot_confusion_matrix(results[name]['confusion_matrix'], name)
                st.pyplot(fig_cm)
                plt.close()
        
        # Bagging SVM Confusion Matrix
        st.subheader("Bagging SVM Confusion Matrix")
        fig_bagging_cm = plot_confusion_matrix(bagging_result['confusion_matrix'], 
                                               'Bagging SVM (Optimized)')
        st.pyplot(fig_bagging_cm)
        plt.close()
        
        # Feature Importance
        st.header("🔬 Step 7: Feature Importance Analysis")
        
        tab1, tab2 = st.tabs(["🌲 Random Forest", "🚀 XGBoost"])
        
        with tab1:
            fig_rf = plot_feature_importance(
                results['Random Forest']['model'],
                features_df.columns.tolist(),
                'Random Forest',
                top_n=15
            )
            if fig_rf:
                st.pyplot(fig_rf)
                plt.close()
        
        with tab2:
            fig_xgb = plot_feature_importance(
                results['XGBoost']['model'],
                features_df.columns.tolist(),
                'XGBoost',
                top_n=15
            )
            if fig_xgb:
                st.pyplot(fig_xgb)
                plt.close()
        
        # Store in session state
        st.session_state['models'] = results
        st.session_state['bagging_svm'] = bagging_result
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = features_df.columns.tolist()
        st.session_state['df_filtered'] = df_filtered
        st.session_state['trained'] = True
        
        st.success("✅ **All models trained successfully!**")
        st.balloons()
    
    # Prediction Interface
    if st.session_state.get('trained', False):
        st.header("🔮 Prediction Interface")
        
        tab1, tab2 = st.tabs(["Single Compound", "Batch Prediction"])
        
        with tab1:
            st.subheader("Predict Activity for New Compound")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                smiles_input = st.text_input(
                    "Enter SMILES string:",
                    placeholder="ClC1=CC=C([C@H]2C3=CC=CC=C3[C@@H](NC)CC2)C=C1Cl",
                    help="Enter the SMILES notation of the compound"
                )
            
            with col2:
                predict_button = st.button("🎯 Predict", type="primary", use_container_width=True)
            
            if predict_button and smiles_input:
                descriptors = calculate_descriptors(smiles_input)
                
                if descriptors:
                    feature_vector = np.array([descriptors[f] for f in st.session_state['feature_names']]).reshape(1, -1)
                    feature_scaled = st.session_state['scaler'].transform(feature_vector)
                    
                    st.subheader("Prediction Results")
                    
                    predictions_data = []
                    
                    # All models predictions
                    for name, result in st.session_state['models'].items():
                        pred = result['model'].predict(feature_scaled)[0]
                        prob = result['model'].predict_proba(feature_scaled)[0][1] if hasattr(result['model'], 'predict_proba') else pred
                        predictions_data.append({
                            'Model': name,
                            'Prediction': '🟢 Active' if pred == 1 else '🔴 Inactive',
                            'Confidence': f"{prob*100:.2f}%"
                        })
                    
                    # Bagging SVM
                    bagging_pred = st.session_state['bagging_svm']['model'].predict(feature_scaled)[0]
                    bagging_prob = st.session_state['bagging_svm']['model'].predict_proba(feature_scaled)[0][1]
                    predictions_data.append({
                        'Model': '⭐ Bagging SVM',
                        'Prediction': '🟢 Active' if bagging_pred == 1 else '🔴 Inactive',
                        'Confidence': f"{bagging_prob*100:.2f}%"
                    })
                    
                    pred_df = pd.DataFrame(predictions_data)
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Consensus
                    active_votes = sum([1 for p in predictions_data if 'Active' in p['Prediction']])
                    total_votes = len(predictions_data)
                    
                    st.markdown("---")
                    if active_votes > total_votes / 2:
                        st.success(f"✅ **Consensus Prediction: ACTIVE** ({active_votes}/{total_votes} models agree)")
                    else:
                        st.warning(f"⚠️ **Consensus Prediction: INACTIVE** ({total_votes-active_votes}/{total_votes} models agree)")
                    
                else:
                    st.error("❌ Invalid SMILES string. Please check and try again.")
        
        with tab2:
            st.subheader("View Predictions for All Dataset Compounds")
            
            if st.button("Generate Full Dataset Predictions"):
                df_results = st.session_state['df_filtered'].copy()
                
                # Use best model
                best_model_name = max(st.session_state['models'].items(), 
                                    key=lambda x: x[1]['accuracy'])[0]
                best_model = st.session_state['models'][best_model_name]['model']
                
                with st.spinner("Generating predictions..."):
                    original_count = len(df_results)
                    features_df_all, df_results_filtered = prepare_features(df_results, show_progress=False)
                    
                    # Validate features before prediction
                    if features_df_all.empty or len(features_df_all.columns) == 0:
                        st.error("❌ Could not extract features for predictions. Please check the dataset.")
                        return
                    
                    if len(df_results_filtered) == 0:
                        st.error("❌ No valid compounds to make predictions for.")
                        return
                    
                    # Use the filtered dataframe which already contains only valid compounds
                    df_results = df_results_filtered.copy()
                    
                    if len(df_results) != original_count:
                        st.warning(f"⚠️ Only {len(df_results)} out of {original_count} compounds had valid features.")
                    
                    X_all_scaled = st.session_state['scaler'].transform(features_df_all.values)
                    
                    predictions = best_model.predict(X_all_scaled)
                    probabilities = best_model.predict_proba(X_all_scaled)[:, 1]
                    
                    df_results['Predicted'] = predictions
                    df_results['Confidence'] = probabilities.round(4)
                    df_results['Correct'] = df_results['activity'] == df_results['Predicted']
                    df_results['Status'] = df_results['Correct'].apply(lambda x: '✅ Correct' if x else '❌ Wrong')
                    
                    # Display results
                    st.dataframe(
                        df_results[['num', 'name', 'activity', 'Predicted', 'Confidence', 'Status']],
                        use_container_width=True
                    )
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    accuracy = (df_results['Correct'].sum() / len(df_results)) * 100
                    correct_active = ((df_results['activity'] == 1) & (df_results['Predicted'] == 1)).sum()
                    total_active = (df_results['activity'] == 1).sum()
                    correct_inactive = ((df_results['activity'] == 0) & (df_results['Predicted'] == 0)).sum()
                    total_inactive = (df_results['activity'] == 0).sum()
                    
                    with col1:
                        st.metric("Overall Accuracy", f"{accuracy:.2f}%")
                    with col2:
                        st.metric("Correct Predictions", f"{df_results['Correct'].sum()}/{len(df_results)}")
                    with col3:
                        active_acc = (correct_active / total_active * 100) if total_active > 0 else 0
                        st.metric("Active Accuracy", f"{active_acc:.2f}%")
                    with col4:
                        inactive_acc = (correct_inactive / total_inactive * 100) if total_inactive > 0 else 0
                        st.metric("Inactive Accuracy", f"{inactive_acc:.2f}%")
                    
                    st.info(f"📊 Model used: **{best_model_name}**")
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="sertraline_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()