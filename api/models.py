import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
import joblib
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split

class SepsisPredictor:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.class_weights = None
        
    def calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced sepsis data"""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        self.class_weights = dict(zip(classes, weights))
        print(f"Class weights: {self.class_weights}")
        return self.class_weights
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None, use_smote=True):
        """Train multiple models for sepsis prediction"""
        
        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Sepsis cases: {y_train.sum()} ({y_train.mean():.2%})")
        
        # Calculate class weights
        self.calculate_class_weights(y_train)
        
        # Handle class imbalance with SMOTE
        if use_smote and y_train.mean() < 0.3:
            print("Applying SMOTE for class imbalance...")
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {X_train_bal.shape}, Sepsis rate: {y_train_bal.mean():.2%}")
        else:
            X_train_bal, y_train_bal = X_train, y_train
        
        # 1. Random Forest
        print("\n1. Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_bal, y_train_bal)
        self.models['random_forest'] = rf
        
        self.feature_importance['random_forest'] = dict(zip(
            X_train.columns, rf.feature_importances_
        ))
        
        # 2. XGBoost
        print("2. Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=self.class_weights[1] / self.class_weights[0] if self.class_weights else 3,
            random_state=42,
            use_label_encoder=False,
            eval_metric='auc'
        )
        xgb.fit(X_train_bal, y_train_bal)
        self.models['xgboost'] = xgb
        
        self.feature_importance['xgboost'] = dict(zip(
            X_train.columns, xgb.feature_importances_
        ))
        
        # 3. Neural Network
        print("3. Training Neural Network...")
        nn_model = self.train_neural_network(X_train_bal, y_train_bal, X_val, y_val)
        self.models['neural_network'] = nn_model
        
        # 4. Gradient Boosting
        print("4. Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train_bal, y_train_bal)
        self.models['gradient_boosting'] = gb
        
        self.feature_importance['gradient_boosting'] = dict(zip(
            X_train.columns, gb.feature_importances_
        ))
        
        return self.models
    
    def train_neural_network(self, X_train, y_train, X_val=None, y_val=None):
        """Train a neural network for sepsis prediction"""
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.Precision(name='precision')]
        )
        
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=0.00001)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
            class_weight=self.class_weights
        )
        
        return model
    
    def predict(self, X, model_name='ensemble', threshold=0.5):
        """Make sepsis predictions"""
        if model_name == 'ensemble':
            probabilities = self.ensemble_predict(X)
        elif model_name in self.models:
            model = self.models[model_name]
            if model_name == 'neural_network':
                probabilities = model.predict(X, verbose=0).flatten()
            else:
                probabilities = model.predict_proba(X)[:, 1]
        else:
            raise ValueError(f"Model {model_name} not found")
        
        predictions = (probabilities > threshold).astype(int)
        return predictions, probabilities
    
    def ensemble_predict(self, X, weights=None):
        """Make ensemble predictions"""
        if weights is None:
            weights = {
                'random_forest': 0.25,
                'xgboost': 0.25,
                'neural_network': 0.30,
                'gradient_boosting': 0.20
            }
        
        predictions = []
        
        for name, weight in weights.items():
            if name in self.models:
                if name == 'neural_network':
                    pred = self.models[name].predict(X, verbose=0).flatten()
                else:
                    pred = self.models[name].predict_proba(X)[:, 1]
                
                predictions.append(pred * weight)
        
        if predictions:
            ensemble_prob = np.sum(predictions, axis=0)
            return ensemble_prob
        else:
            raise ValueError("No models available for ensemble prediction")
    
    def evaluate_models(self, X_test, y_test, threshold=0.5):
        """Evaluate all models"""
        results = {}
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            if name == 'neural_network':
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            y_pred = (y_pred_proba > threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            results[name] = {
                'auc': roc_auc_score(y_test, y_pred_proba),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
            }
            
            print(f"  AUC: {results[name]['auc']:.4f}")
            print(f"  Recall: {results[name]['recall']:.4f}")
            print(f"  Specificity: {results[name]['specificity']:.4f}")
        
        # Ensemble evaluation
        print("\nEvaluating Ensemble Model...")
        y_pred_proba_ensemble = self.ensemble_predict(X_test)
        y_pred_ensemble = (y_pred_proba_ensemble > threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ensemble).ravel()
        
        results['ensemble'] = {
            'auc': roc_auc_score(y_test, y_pred_proba_ensemble),
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble),
            'recall': recall_score(y_test, y_pred_ensemble),
            'f1': f1_score(y_test, y_pred_ensemble),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        
        print(f"  AUC: {results['ensemble']['auc']:.4f}")
        print(f"  Recall: {results['ensemble']['recall']:.4f}")
        print(f"  Specificity: {results['ensemble']['specificity']:.4f}")
        
        return results
    
    def find_optimal_threshold(self, X_val, y_val):
        """Find optimal threshold for sepsis prediction"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred_proba = self.ensemble_predict(X_val)
            y_pred = (y_pred_proba > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Weighted score favoring recall (don't miss sepsis cases)
            score = 0.7 * recall + 0.3 * specificity
            
            results.append({
                'threshold': float(threshold),  # Convert numpy to Python float
                'recall': float(recall),        # Convert numpy to Python float
                'specificity': float(specificity), # Convert numpy to Python float
                'score': float(score)           # Convert numpy to Python float
            })
        
        optimal = max(results, key=lambda x: x['score'])
        
        print(f"\nOptimal threshold: {optimal['threshold']:.3f}")
        print(f"Recall: {optimal['recall']:.3f}, Specificity: {optimal['specificity']:.3f}")
        
        return optimal['threshold'], results
    
    def save_models(self, directory='models'):
        """Save trained models - FIXED VERSION"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(f'{directory}/{name}_model.h5')
            else:
                joblib.dump(model, f'{directory}/{name}_model.pkl')
        
        # Save metadata - FIXED: Convert numpy types to Python native types
        import json
        
        # Convert feature importance to serializable format
        serializable_importance = {}
        for model_name, importance_dict in self.feature_importance.items():
            serializable_importance[model_name] = {
                str(feature): float(importance_value)  # Convert numpy to Python types
                for feature, importance_value in importance_dict.items()
            }
        
        # Convert class weights to serializable format
        serializable_weights = None
        if self.class_weights is not None:
            serializable_weights = {
                int(key): float(value)  # Convert numpy int/float to Python types
                for key, value in self.class_weights.items()
            }
        
        # Save the serializable data
        with open(f'{directory}/model_metadata.json', 'w') as f:
            json.dump({
                'feature_importance': serializable_importance,
                'class_weights': serializable_weights
            }, f, indent=2)
        
        print(f"Models saved to {directory}/")
    
    def load_models(self, directory='models'):
        """Load trained models - UPDATED VERSION"""
        import os
        
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl',
            'neural_network': 'neural_network_model.h5'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                if filename.endswith('.h5'):
                    self.models[name] = load_model(filepath)
                else:
                    self.models[name] = joblib.load(filepath)
                print(f"Loaded {name} model")
        
        # Load metadata
        metadata_file = os.path.join(directory, 'model_metadata.json')
        if os.path.exists(metadata_file):
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.feature_importance = metadata.get('feature_importance', {})
                
                # Convert class weights back from JSON (they're already Python types)
                class_weights_json = metadata.get('class_weights', None)
                if class_weights_json is not None:
                    # JSON keys are strings, convert back to int
                    self.class_weights = {
                        int(k): float(v) for k, v in class_weights_json.items()
                    }