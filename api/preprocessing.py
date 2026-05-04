import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SepsisDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.label_encoder = LabelEncoder()
        
        # Feature mapping for sepsis prediction
        self.feature_mapping = {
            'PRG': 'Pregnancies_or_Respiratory_Rate',
            'PL': 'Platelets_or_Lactate',
            'PR': 'Pulse_Rate_or_BP',
            'SK': 'Serum_Potassium',
            'TS': 'Temperature_or_Score',
            'M11': 'Mean_Arterial_Pressure',
            'BD2': 'Base_Deficit_Bicarbonate',
            'Age': 'Age',
            'Insurance': 'Insurance_Status'
        }
        
        self.feature_names = list(self.feature_mapping.keys())
        
        # Medical validation ranges
        self.medical_ranges = {
            'PRG': (0, 50),
            'PL': (0, 1000),
            'PR': (0, 250),
            'SK': (0, 10),
            'TS': (30, 45),
            'M11': (40, 150),
            'BD2': (-20, 30),
            'Age': (0, 120),
            'Insurance': (0, 1)
        }
    
    def load_and_prepare_data(self, train_path, test_path):
        """Load train and test datasets"""
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            print("=" * 60)
            print("SEPSIS DATASET ANALYSIS")
            print("=" * 60)
            print(f"Training data shape: {train_df.shape}")
            print(f"Test data shape: {test_df.shape}")
            
            if 'Sepssis' in train_df.columns:
                sepsis_counts = train_df['Sepssis'].value_counts()
                print(f"\nClass distribution: {sepsis_counts.to_dict()}")
                print(f"Sepsis rate: {sepsis_counts.get('Positive', 0)/len(train_df):.2%}")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df, is_train=True):
        """Preprocess the sepsis data"""
        df_processed = df.copy()
        
        # Handle target variable
        if is_train and 'Sepssis' in df_processed.columns:
            df_processed['SepsisLabel'] = df_processed['Sepssis'].map({
                'Positive': 1,
                'Negative': 0
            })
            df_processed['SepsisLabel'] = df_processed['SepsisLabel'].fillna(0)
            
            X = df_processed[self.feature_names]
            y = df_processed['SepsisLabel']
        else:
            X = df_processed[self.feature_names]
            y = None
        
        # Handle missing values
        X_imputed = X.copy()
        
        # Convert 0 to NaN for features where 0 is medically impossible
        features_with_zero_as_missing = ['PL', 'PR', 'SK', 'TS', 'M11', 'BD2']
        for feature in features_with_zero_as_missing:
            if feature in X_imputed.columns:
                X_imputed[feature] = X_imputed[feature].replace(0, np.nan)
        
        # Impute missing values
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_imputed) if is_train else self.imputer.transform(X_imputed),
            columns=X.columns
        )
        
        # Normalize features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed) if is_train else self.scaler.transform(X_imputed),
            columns=X.columns
        )
        
        return X_scaled, y, df_processed
    
    def create_sepsis_features(self, df):
        """Create sepsis-specific derived features"""
        df_features = df.copy()
        
        # SIRS Criteria features
        if 'TS' in df_features.columns:
            df_features['sirs_temp'] = ((df_features['TS'] > 38) | (df_features['TS'] < 36)).astype(int)
        
        if 'PR' in df_features.columns:
            df_features['sirs_hr'] = (df_features['PR'] > 90).astype(int)
        
        if 'PL' in df_features.columns:
            df_features['elevated_lactate'] = (df_features['PL'] > 2.0).astype(int)
            df_features['severe_lactate'] = (df_features['PL'] > 4.0).astype(int)
        
        if 'SK' in df_features.columns:
            df_features['electrolyte_imbalance'] = ((df_features['SK'] > 5.5) | (df_features['SK'] < 3.5)).astype(int)
        
        if 'M11' in df_features.columns:
            df_features['hypotension'] = (df_features['M11'] < 65).astype(int)
        
        if 'Age' in df_features.columns:
            df_features['elderly_risk'] = (df_features['Age'] > 65).astype(int)
        
        # Create sepsis risk score
        risk_components = [
            'sirs_temp', 'sirs_hr', 'elevated_lactate', 
            'electrolyte_imbalance', 'hypotension', 'elderly_risk'
        ]
        
        existing_components = [col for col in risk_components if col in df_features.columns]
        if existing_components:
            df_features['sepsis_risk_score'] = df_features[existing_components].sum(axis=1)
        
        # qSOFA criteria
        qsofa_components = []
        
        if 'M11' in df_features.columns:
            df_features['qsofa_hypotension'] = (df_features['M11'] < 100).astype(int)
            qsofa_components.append('qsofa_hypotension')
        
        if 'PR' in df_features.columns:
            df_features['qsofa_tachypnea_proxy'] = (df_features['PR'] > 90).astype(int)
            qsofa_components.append('qsofa_tachypnea_proxy')
        
        if 'Age' in df_features.columns:
            df_features['qsofa_age_risk'] = (df_features['Age'] > 75).astype(int)
            qsofa_components.append('qsofa_age_risk')
        
        if qsofa_components:
            df_features['qsofa_score'] = df_features[qsofa_components].sum(axis=1)
            df_features['high_qsofa'] = (df_features['qsofa_score'] >= 2).astype(int)
        
        return df_features
    
    def get_feature_description(self, feature_code):
        """Get medical description of each feature"""
        descriptions = {
            'PRG': 'Possibly: Pregnancies or Respiratory Rate parameter',
            'PL': 'Likely: Platelet count (critical for sepsis) OR Lactate level',
            'PR': 'Possibly: Pulse Rate (tachycardia in sepsis) or Blood Pressure',
            'SK': 'Serum Potassium (electrolyte imbalance in sepsis)',
            'TS': 'Possibly: Temperature (fever/hypothermia in sepsis)',
            'M11': 'Mean Arterial Pressure (hypotension in septic shock)',
            'BD2': 'Base Deficit or Bicarbonate (metabolic acidosis in sepsis)',
            'Age': 'Patient age (higher risk in elderly)',
            'Insurance': 'Insurance status (access to care)'
        }
        return descriptions.get(feature_code, 'Unknown parameter')
    
    def get_sepsis_risk_interpretation(self, feature_values):
        """Interpret feature values for sepsis risk"""
        interpretations = []
        
        if 'PL' in feature_values:
            pl_val = feature_values['PL']
            if pl_val > 4.0:
                interpretations.append(f"🚨 Critically elevated lactate/platelets ({pl_val:.1f})")
            elif pl_val > 2.0:
                interpretations.append(f"⚠️ Elevated lactate/platelets ({pl_val:.1f})")
        
        if 'M11' in feature_values:
            map_val = feature_values['M11']
            if map_val < 65:
                interpretations.append(f"🚨 Severe hypotension (MAP: {map_val:.0f} mmHg)")
            elif map_val < 100:
                interpretations.append(f"⚠️ Low blood pressure (MAP: {map_val:.0f} mmHg)")
        
        if 'PR' in feature_values:
            pr_val = feature_values['PR']
            if pr_val > 120:
                interpretations.append(f"⚠️ Severe tachycardia ({pr_val:.0f} bpm)")
            elif pr_val > 90:
                interpretations.append(f"⚠️ Elevated heart rate ({pr_val:.0f} bpm)")
        
        if 'Age' in feature_values:
            age_val = feature_values['Age']
            if age_val > 75:
                interpretations.append(f"⚠️ Advanced age ({age_val:.0f} years)")
        
        return interpretations
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and validation sets"""
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state, stratify=y)