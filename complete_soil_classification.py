# complete_soil_classification.py - Full ML System with SVM, RF, and CNN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime

class SoilDataPreprocessor:
    """Handle data preprocessing including missing values and normalization"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = ['sand_percent', 'silt_percent', 'clay_percent', 
                             'ph', 'organic_content', 'nitrogen', 'phosphorus', 'potassium']
        
    def load_data(self, filepath):
        """Load soil chemistry dataset"""
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values using appropriate strategies"""
        print("\n=== Handling Missing Values ===")
        print(f"Missing values before:\n{df.isnull().sum()}")
        
        # For numerical features: use median imputation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"Filled {col} with median: {median_value:.2f}")
        
        # For categorical: use mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                print(f"Filled {col} with mode: {mode_value}")
        
        print(f"\nMissing values after:\n{df.isnull().sum()}")
        return df
    
    def normalize_features(self, X_train, X_test):
        """Normalize features using StandardScaler"""
        print("\n=== Normalizing Features ===")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training data - Mean: {X_train_scaled.mean():.4f}, Std: {X_train_scaled.std():.4f}")
        print(f"Test data - Mean: {X_test_scaled.mean():.4f}, Std: {X_test_scaled.std():.4f}")
        
        return X_train_scaled, X_test_scaled
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using oversampling"""
        print("\n=== Handling Class Imbalance ===")
        df = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        df['soil_type'] = y
        
        # Count samples per class
        class_counts = df['soil_type'].value_counts()
        print(f"Original class distribution:\n{class_counts}")
        
        # Find maximum class size
        max_size = class_counts.max()
        
        # Oversample minority classes
        dfs = []
        for soil_type in class_counts.index:
            df_class = df[df['soil_type'] == soil_type]
            df_resampled = resample(df_class, 
                                   replace=True,
                                   n_samples=max_size,
                                   random_state=42)
            dfs.append(df_resampled)
        
        df_balanced = pd.concat(dfs)
        
        print(f"\nBalanced class distribution:\n{df_balanced['soil_type'].value_counts()}")
        
        X_balanced = df_balanced.drop('soil_type', axis=1).values
        y_balanced = df_balanced['soil_type'].values
        
        return X_balanced, y_balanced
    
    def encode_labels(self, y):
        """Encode soil type labels to numerical values"""
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"\nLabel encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        return y_encoded


class SoilClassificationModels:
    """Train and evaluate different ML models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train Support Vector Machine with hyperparameter tuning"""
        print("\n" + "="*60)
        print("TRAINING SUPPORT VECTOR MACHINE (SVM)")
        print("="*60)
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        print("Performing GridSearchCV...")
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', 
                                   n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Train final model
        best_svm = grid_search.best_estimator_
        y_pred = best_svm.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.models['SVM'] = best_svm
        self.results['SVM'] = {
            'accuracy': accuracy,
            'best_params': grid_search.best_params_,
            'predictions': y_pred,
            'y_test': y_test
        }
        
        return best_svm
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest with hyperparameter tuning"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        print("Performing GridSearchCV...")
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy',
                                   n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Train final model
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': ['sand', 'silt', 'clay', 'pH', 'organic', 'N', 'P', 'K'][:X_train.shape[1]],
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        self.models['RandomForest'] = best_rf
        self.results['RandomForest'] = {
            'accuracy': accuracy,
            'best_params': grid_search.best_params_,
            'feature_importance': feature_importance.to_dict(),
            'predictions': y_pred,
            'y_test': y_test
        }
        
        return best_rf
    
    def compare_models(self, label_encoder):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results.keys()]
        }).sort_values('Accuracy', ascending=False)
        
        print(comparison.to_string(index=False))
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.bar(comparison['Model'], comparison['Accuracy'])
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.ylim([0.7, 1.0])
        for i, v in enumerate(comparison['Accuracy']):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300)
        print("\nComparison plot saved as 'model_comparison.png'")
        
        return comparison
    
    def plot_confusion_matrices(self, label_encoder):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = confusion_matrix(results['y_test'], results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], 
                       xticklabels=label_encoder.classes_,
                       yticklabels=label_encoder.classes_,
                       cmap='Blues')
            axes[idx].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.4f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300)
        print("Confusion matrices saved as 'confusion_matrices.png'")
    
    def save_models(self):
        """Save all trained models"""
        for model_name, model in self.models.items():
            filename = f'{model_name.lower()}_model.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} model as '{filename}'")


def generate_synthetic_dataset(n_samples=1000):
    """Generate synthetic soil chemistry dataset for demonstration"""
    print("Generating synthetic soil chemistry dataset...")
    
    np.random.seed(42)
    
    soil_types = ['Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil',
                  'Mountain_Soil', 'Red_Soil', 'Yellow_Soil']
    
    data = []
    
    # Define soil characteristics for each type
    soil_profiles = {
        'Alluvial_Soil': {'sand': (30, 50), 'silt': (30, 50), 'clay': (10, 25), 
                          'ph': (6.5, 8.0), 'organic': (2.0, 4.0)},
        'Arid_Soil': {'sand': (70, 90), 'silt': (5, 15), 'clay': (5, 15),
                      'ph': (7.5, 9.0), 'organic': (0.5, 1.5)},
        'Black_Soil': {'sand': (20, 35), 'silt': (20, 35), 'clay': (40, 60),
                       'ph': (7.0, 8.5), 'organic': (3.0, 5.5)},
        'Laterite_Soil': {'sand': (35, 55), 'silt': (15, 30), 'clay': (25, 45),
                          'ph': (5.0, 6.5), 'organic': (1.5, 3.0)},
        'Mountain_Soil': {'sand': (40, 60), 'silt': (20, 35), 'clay': (15, 30),
                          'ph': (5.5, 7.0), 'organic': (4.0, 7.0)},
        'Red_Soil': {'sand': (50, 70), 'silt': (15, 25), 'clay': (15, 30),
                     'ph': (5.5, 7.5), 'organic': (1.0, 2.5)},
        'Yellow_Soil': {'sand': (45, 65), 'silt': (20, 35), 'clay': (15, 30),
                        'ph': (5.0, 6.5), 'organic': (1.5, 3.5)}
    }
    
    samples_per_class = n_samples // len(soil_types)
    
    for soil_type in soil_types:
        profile = soil_profiles[soil_type]
        
        for _ in range(samples_per_class):
            # Generate texture percentages (must sum to 100)
            sand = np.random.uniform(*profile['sand'])
            remaining = 100 - sand
            clay_max = min(profile['clay'][1], remaining - profile['silt'][0])
            clay = np.random.uniform(profile['clay'][0], clay_max)
            silt = 100 - sand - clay
            
            # Generate other properties
            ph = np.random.uniform(*profile['ph'])
            organic = np.random.uniform(*profile['organic'])
            
            # Add NPK values
            nitrogen = np.random.uniform(20, 300)
            phosphorus = np.random.uniform(10, 100)
            potassium = np.random.uniform(50, 400)
            
            data.append({
                'sand_percent': sand,
                'silt_percent': silt,
                'clay_percent': clay,
                'ph': ph,
                'organic_content': organic,
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'soil_type': soil_type
            })
    
    df = pd.DataFrame(data)
    
    # Randomly add some missing values (5% missing rate)
    for col in df.columns[:-1]:  # Don't add missing to soil_type
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Generated {len(df)} samples across {len(soil_types)} soil types")
    print(f"\nDataset preview:")
    print(df.head())
    print(f"\nClass distribution:")
    print(df['soil_type'].value_counts())
    
    return df


def main():
    """Main execution pipeline"""
    print("="*60)
    print("COMPREHENSIVE SOIL CLASSIFICATION SYSTEM")
    print("With SVM, Random Forest, and Data Preprocessing")
    print("="*60)
    
    # Step 1: Generate/Load Dataset
    df = generate_synthetic_dataset(n_samples=1000)
    df.to_csv('soil_chemistry_dataset.csv', index=False)
    print("\nDataset saved as 'soil_chemistry_dataset.csv'")
    
    # Step 2: Initialize Preprocessor
    preprocessor = SoilDataPreprocessor()
    
    # Step 3: Handle Missing Values
    df = preprocessor.handle_missing_values(df)
    
    # Step 4: Split Features and Target
    X = df.drop('soil_type', axis=1).values
    y = df['soil_type'].values
    
    # Step 5: Encode Labels
    y_encoded = preprocessor.encode_labels(y)
    
    # Step 6: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 7: Handle Class Imbalance (on training data only)
    X_train_balanced, y_train_balanced = preprocessor.handle_class_imbalance(
        X_train, y_train
    )
    
    # Step 8: Normalize Features
    X_train_normalized, X_test_normalized = preprocessor.normalize_features(
        X_train_balanced, X_test
    )
    
    # Step 9: Train Models
    classifier = SoilClassificationModels()
    
    # Train SVM
    svm_model = classifier.train_svm(
        X_train_normalized, y_train_balanced,
        X_test_normalized, y_test
    )
    
    # Train Random Forest
    rf_model = classifier.train_random_forest(
        X_train_normalized, y_train_balanced,
        X_test_normalized, y_test
    )
    
    # Step 10: Compare Models
    comparison = classifier.compare_models(preprocessor.label_encoder)
    
    # Step 11: Visualize Results
    classifier.plot_confusion_matrices(preprocessor.label_encoder)
    
    # Step 12: Save Models
    classifier.save_models()
    
    # Save preprocessor
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Saved preprocessor as 'preprocessor.pkl'")
    
    # Save results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(df),
        'features': preprocessor.feature_names,
        'soil_types': preprocessor.label_encoder.classes_.tolist(),
        'model_results': {
            model: {
                'accuracy': float(results['accuracy']),
                'best_params': results['best_params']
            }
            for model, results in classifier.results.items()
        }
    }
    
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nTraining summary saved as 'training_summary.json'")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("- soil_chemistry_dataset.csv")
    print("- svm_model.pkl")
    print("- randomforest_model.pkl")
    print("- preprocessor.pkl")
    print("- model_comparison.png")
    print("- confusion_matrices.png")
    print("- training_summary.json")


if __name__ == "__main__":
    main()