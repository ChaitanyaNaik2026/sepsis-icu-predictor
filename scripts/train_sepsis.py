import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import json  # Add this import

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.preprocessing import SepsisDataPreprocessor
from api.models import SepsisPredictor

def main():
    """Main training pipeline"""
    
    print("=" * 70)
    print("SEPSIS PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    # Initialize
    preprocessor = SepsisDataPreprocessor()
    
    # Step 1: Load data
    print("\n1. Loading data...")
    train_df, test_df = preprocessor.load_and_prepare_data(
        'data/Paitients_Files_Train.csv',
        'data/Paitients_Files_Test.csv'
    )
    
    # Step 2: Preprocess
    print("\n2. Preprocessing...")
    X_train, y_train, _ = preprocessor.preprocess_data(train_df, is_train=True)
    print(f"   Samples: {len(X_train)}")
    print(f"   Sepsis cases: {y_train.sum()} ({y_train.mean():.2%})")
    
    # Step 3: Create sepsis features
    print("\n3. Creating sepsis features...")
    X_train_enhanced = preprocessor.create_sepsis_features(X_train)
    print(f"   Original features: {X_train.shape[1]}")
    print(f"   Enhanced features: {X_train_enhanced.shape[1]}")
    
    # Step 4: Split data
    print("\n4. Splitting data...")
    X_train_split, X_val, y_train_split, y_val = preprocessor.split_data(
        X_train_enhanced, y_train, test_size=0.2
    )
    
    # Step 5: Train models
    print("\n5. Training models...")
    predictor = SepsisPredictor()
    predictor.train_models(X_train_split, y_train_split, X_val, y_val)
    
    # Step 6: Evaluate
    print("\n6. Evaluating...")
    results = predictor.evaluate_models(X_val, y_val)
    
    # Step 7: Find optimal threshold
    print("\n7. Finding optimal threshold...")
    optimal_threshold, _ = predictor.find_optimal_threshold(X_val, y_val)
    
    # Step 8: Test on hold-out data
    print("\n8. Testing on hold-out data...")
    X_test, _, _ = preprocessor.preprocess_data(test_df, is_train=False)
    X_test_enhanced = preprocessor.create_sepsis_features(X_test)
    test_probabilities = predictor.ensemble_predict(X_test_enhanced)
    test_predictions = (test_probabilities > optimal_threshold).astype(int)
    
    # Step 9: Save models
    print("\n9. Saving models...")
    os.makedirs('models', exist_ok=True)
    predictor.save_models('models')
    
    # Create report - FIXED VERSION
    report = {
        "timestamp": str(pd.Timestamp.now()),
        "training_summary": {
            "samples": int(len(X_train)),
            "sepsis_cases": int(y_train.sum()),
            "sepsis_rate": float(y_train.mean()),
            "features": int(X_train_enhanced.shape[1])
        },
        "model_performance": {
            name: {
                'auc': float(results[name]['auc']),
                'recall': float(results[name]['recall']),
                'specificity': float(results[name]['specificity']),
                'accuracy': float(results[name]['accuracy']),
                'precision': float(results[name]['precision']),
                'f1': float(results[name]['f1']),
                'confusion_matrix': {
                    'tn': int(results[name]['confusion_matrix']['tn']),
                    'fp': int(results[name]['confusion_matrix']['fp']),
                    'fn': int(results[name]['confusion_matrix']['fn']),
                    'tp': int(results[name]['confusion_matrix']['tp'])
                }
            }
            for name in results.keys()
        },
        "optimal_threshold": float(optimal_threshold),
        "test_predictions": {
            "total": int(len(test_df)),
            "predicted_sepsis": int(test_predictions.sum()),
            "predicted_rate": float(test_probabilities.mean()),
            "average_risk": float(test_probabilities.mean())
        }
    }
    
    with open("models/training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Display summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 Performance:")
    print(f"  Ensemble AUC: {results['ensemble']['auc']:.4f}")
    print(f"  Recall: {results['ensemble']['recall']:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    
    print(f"\n📈 Test Data:")
    print(f"  Patients: {len(test_df)}")
    print(f"  Predicted sepsis: {test_predictions.sum()} ({test_probabilities.mean():.1%})")
    
    print(f"\n🚀 To start:")
    print(f"  API: uvicorn api.main:app --reload --port 8000")
    print(f"  UI: streamlit run app/streamlit_app.py")
    print("=" * 70)

if __name__ == "__main__":
    main()