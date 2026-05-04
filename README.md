# 🏥 Sepsis ICU Predictor

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)
![ML](https://img.shields.io/badge/ML-Ensemble-red)
![API](https://img.shields.io/badge/API-FastAPI-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-yellow)

An AI-powered system for **early detection of sepsis in ICU patients** using a machine learning ensemble architecture. The platform predicts sepsis risk from patient vitals and delivers **clinical recommendations for timely intervention** through both a REST API and an interactive web dashboard.

---

## 🔁 System Architecture

The system follows a modular ML pipeline with ensemble inference and clinical decision support:

**Pipeline Flow:**

1. Patient Data Input
2. Data Preprocessing
3. Feature Engineering
4. ML Ensemble

   * Random Forest
   * XGBoost
   * Neural Network
   * Gradient Boosting
5. Weighted Voting
6. Risk Prediction (0–1 Probability)
7. Clinical Analysis
8. Risk Stratification
9. Actionable Recommendations

This design prioritizes **high recall** to minimize missed sepsis cases while maintaining clinical interpretability.

---

## 🎯 Features

* 🤖 **Multi-Model Ensemble**
  Combines Random Forest, XGBoost, Neural Network, and Gradient Boosting using weighted voting

* ⚕️ **Clinical Decision Support**
  Generates actionable recommendations based on predicted risk levels

* 🌐 **Web Dashboard**
  Streamlit-based real-time patient risk assessment interface

* 🔌 **REST API**
  FastAPI backend for hospital system integration

* 📊 **Real-Time Analytics**
  Visualization of risk factors and model performance

* ⚡ **High Recall Optimization**
  Tuned to detect up to 100% of sepsis cases (minimizing false negatives)

---

## 📊 Model Performance

| Model             | AUC       | Recall    | Specificity | Notes          |
| ----------------- | --------- | --------- | ----------- | -------------- |
| Ensemble          | 0.82–0.85 | 0.80–0.85 | 0.75–0.80   | Primary model  |
| Neural Network    | 0.83–0.84 | 0.83–0.88 | 0.66–0.72   | Best recall    |
| Random Forest     | 0.82–0.83 | 0.66–0.67 | 0.73–0.74   | Balanced       |
| XGBoost           | 0.81–0.82 | 0.66–0.67 | 0.73–0.74   | Fast inference |
| Gradient Boosting | 0.81–0.82 | 0.54–0.55 | 0.80–0.81   | Conservative   |

**Optimal Threshold:** `0.15–0.20`
Configured for **100% recall** with **43–54% specificity**

---

## 📁 Project Structure

```
sepsis-prediction-system/
├── api/                          # FastAPI backend
│   ├── __init__.py
│   ├── main.py                 # API endpoints
│   ├── models.py              # ML model loading & inference
│   └── preprocessing.py      # Data preprocessing & validation
│
├── app/                       # Streamlit frontend
│   └── streamlit_app.py     # Web dashboard
│
├── data/                     # Datasets
│   ├── Patients_Files_Train.csv
│   └── Patients_Files_Test.csv
│
├── models/                  # Trained model artifacts
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── neural_network_model.h5
│   ├── gradient_boosting_model.pkl
│   └── model_metadata.json
│
├── scripts/                # Training pipeline
│   └── train_sepsis.py
│
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* pip package manager
* Minimum 4GB RAM

---

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/sepsis-prediction-system.git
cd sepsis-prediction-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place your datasets in the `data/` directory:

* `Patients_Files_Train.csv`
* `Patients_Files_Test.csv`

---

### Train the Models

```bash
python scripts/train_sepsis.py
```

---

### Start the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

---

### Start the Web Dashboard

Open a new terminal:

```bash
streamlit run app/streamlit_app.py
```

---

### Access the System

* API: `http://localhost:8000`
* API Docs: `http://localhost:8000/docs`
* Web UI: `http://localhost:8501`

---

## 🏥 Clinical Features Analyzed

| Feature   | Description                    | Medical Significance           |
| --------- | ------------------------------ | ------------------------------ |
| PRG       | Respiratory Rate / Pregnancies | Respiratory distress indicator |
| PL        | Platelets / Lactate            | Coagulation & perfusion        |
| PR        | Pulse Rate                     | Sepsis tachycardia marker      |
| SK        | Serum Potassium                | Electrolyte imbalance          |
| TS        | Temperature                    | Fever / Hypothermia            |
| M11       | Mean Arterial Pressure         | Hypotension / Shock            |
| BD2       | Base Deficit / Bicarbonate     | Metabolic acidosis             |
| Age       | Patient Age                    | Elderly risk factor            |
| Insurance | Insurance Status               | Access to care                 |

---

## 🖥️ Web Dashboard Usage

1. Open `http://localhost:8501`
2. Navigate to **📊 Patient Assessment**
3. Enter patient vitals
4. Click **Assess Sepsis Risk**

### Output Includes

* Risk level (CRITICAL / HIGH / MODERATE / LOW / VERY LOW)
* Probability score
* Clinical indicators
* Immediate actions
* Monitoring recommendations

---

## 📡 API Usage

### Single Prediction Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "patient_data": {
            "PRG": 1.0,
            "PL": 120.0,
            "PR": 80.0,
            "SK": 4.0,
            "TS": 37.0,
            "M11": 90.0,
            "BD2": 0.0,
            "Age": 45.0,
            "Insurance": 1.0
        }
    }
)

print(response.json())
```

### Example Response

```json
{
  "prediction_time": "2026-01-22T22:40:15.123456",
  "sepsis_probability": 0.1876,
  "risk_level": "LOW",
  "confidence": 15.0,
  "clinical_indicators": [],
  "immediate_actions": ["Continue routine monitoring", "Maintain vigilance"],
  "monitoring_recommendations": ["Standard monitoring"]
}
```

---

## 🔧 Customization

### Adjusting Sensitivity

The system is optimized for **high recall**. To balance recall and specificity:

Edit `api/models.py`:

```python
# Current
score = 0.7 * recall + 0.3 * specificity

# Balanced
score = 0.5 * recall + 0.5 * specificity
```

---

### Adding New Features

Update `api/preprocessing.py`:

```python
self.feature_mapping['NEW_FEATURE'] = 'Description'
self.medical_ranges['NEW_FEATURE'] = (min_val, max_val)
```

---

## 📈 Model Training Pipeline

### Training Flow

1. Data Loading
2. Missing Value Handling
3. Feature Normalization
4. Feature Engineering
5. Model Training (4 models)
6. Ensemble Construction
7. Performance Evaluation
8. Model Serialization

### Retrain Models

```bash
python scripts/train_sepsis.py
```

---

## 🐛 Troubleshooting

### Common Issues

**Import Error**

```
ModuleNotFoundError: No module named 'preprocessing'
```

**Solution:**
Ensure the file is named `preprocessing.py` and use relative imports:

```python
from .preprocessing import preprocess_data
```

---

**Port in Use**

```bash
uvicorn api.main:app --reload --port 8001
streamlit run app/streamlit_app.py --server.port 8502
```

---

**Models Not Loading**

```bash
python scripts/train_sepsis.py
```

---

## 🚨 Clinical Disclaimer

⚠️ **IMPORTANT MEDICAL DISCLAIMER**

This system is for **RESEARCH AND EDUCATIONAL PURPOSES ONLY**.

* NOT for clinical diagnosis
* NOT a replacement for medical professionals
* ALWAYS validate predictions clinically

The authors assume no responsibility for medical decisions made using this system.

---

## 📚 Dataset Information

* Training samples: 599 patients
* Test samples: 169 patients
* Sepsis prevalence: 34.72%
* Raw features: 9
* Engineered features: 22

---

## 👥 Contributing

1. Fork the repository
2. Create a feature branch

   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit changes

   ```bash
   git commit -m "Add AmazingFeature"
   ```
4. Push branch

   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgments

* Sepsis research dataset providers
* scikit-learn, XGBoost, TensorFlow
* FastAPI & Streamlit communities
* Medical professionals for clinical insights

---

## 📞 Contact

For issues or support, please open a GitHub issue.

---

**Developed for educational and research purposes**
**Last Updated: January 2026**
