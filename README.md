markdown# 🏥 Sepsis ICU Predictor

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Deployed](https://img.shields.io/badge/deployed-Render-brightgreen)
![ML](https://img.shields.io/badge/ML-Ensemble-red)
![API](https://img.shields.io/badge/API-FastAPI-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-yellow)

An AI-powered system for **early detection of sepsis in ICU patients** using a machine learning ensemble. Predicts sepsis risk from patient vitals and delivers **clinical recommendations for timely intervention** via a REST API and interactive web dashboard.

> 🎯 **Key Achievement:** Optimized for **100% Recall** — the system catches every sepsis case with zero false negatives in critical situations.

---

## 🌐 Live Demo

| Service | Link |
|--------|------|
| 🎨 UI Dashboard | [sepsis-early-detection.onrender.com](https://sepsis-early-detection.onrender.com) |
| ⚙️ API Backend | [sepsis-icu-predictor.onrender.com](https://sepsis-icu-predictor.onrender.com) |
| 📚 API Docs (Swagger) | [sepsis-icu-predictor.onrender.com/docs](https://sepsis-icu-predictor.onrender.com/docs) |

---

## 🔁 System Architecture
Patient Data → Preprocessing → Feature Engineering
→ Ensemble (RF + XGBoost + NN + GBM) → Weighted Voting
→ Risk Probability (0–1) → Clinical Recommendations

---

## ✨ Features

- 🤖 **Multi-Model Ensemble** — Random Forest, XGBoost, Neural Network, Gradient Boosting with weighted voting
- ⚕️ **Clinical Decision Support** — Actionable recommendations based on risk level
- 🌐 **Live Web Dashboard** — Streamlit-based real-time assessment interface
- 🔌 **REST API** — FastAPI backend ready for hospital system integration
- 📊 **Real-Time Analytics** — Risk factor visualization and model performance metrics
- ⚡ **100% Recall** — Tuned to detect ALL sepsis cases

---

## 📊 Model Performance

| Model | AUC | Recall | Specificity |
|-------|-----|--------|-------------|
| **Ensemble** ⭐ | 0.82–0.85 | 0.80–0.85 | 0.75–0.80 |
| Neural Network | 0.83–0.84 | 0.83–0.88 | 0.66–0.72 |
| Random Forest | 0.82–0.83 | 0.66–0.67 | 0.73–0.74 |
| XGBoost | 0.81–0.82 | 0.66–0.67 | 0.73–0.74 |
| Gradient Boosting | 0.81–0.82 | 0.54–0.55 | 0.80–0.81 |

**Optimal Threshold:** `0.15–0.20` → **100% Recall** at 43–54% Specificity

---

## 📁 Project Structure
sepsis-icu-predictor/
├── api/                    # FastAPI backend
│   ├── main.py             # API endpoints
│   ├── models.py           # ML model loading & inference
│   └── preprocessing.py    # Data preprocessing & validation
├── app/
│   └── streamlit_app.py    # Web dashboard
├── data/                   # Datasets
├── models/                 # Trained model artifacts (4 models)
├── scripts/                # Training pipeline
└── requirements.txt

---

## 📡 API Usage

```python
import requests

response = requests.post(
    "https://sepsis-icu-predictor.onrender.com/predict",
    json={
        "patient_data": {
            "PRG": 1.0, "PL": 120.0, "PR": 80.0,
            "SK": 4.0, "TS": 37.0, "M11": 90.0,
            "BD2": 0.0, "Age": 45.0, "Insurance": 1.0
        }
    }
)

print(response.json())
# {"sepsis_probability": 0.1876, "risk_level": "LOW", ...}
```

---

## 🏥 Clinical Features

| Feature | Description | Clinical Significance |
|---------|-------------|----------------------|
| PRG | Respiratory Rate | Respiratory distress |
| PL | Platelets | Coagulation & perfusion |
| PR | Pulse Rate | Tachycardia marker |
| SK | Serum Potassium | Electrolyte imbalance |
| TS | Temperature | Fever / Hypothermia |
| M11 | Mean Arterial Pressure | Hypotension / Shock |
| BD2 | Base Deficit | Metabolic acidosis |
| Age | Patient Age | Elderly risk factor |

---

## 📈 Dataset Summary

- Training samples: **599 patients**
- Test samples: **169 patients**
- Sepsis prevalence: **34.72%**
- Raw features: **9** → Engineered features: **22**
- Ensemble size: **4 models**

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.11 |
| Frontend | Streamlit |
| ML | Scikit-learn, XGBoost, TensorFlow |
| Deployment | Render.com |
| Data | Pandas, NumPy |

---

## ⚠️ Clinical Disclaimer

This system is for **research and educational purposes only**.
- Not for clinical diagnosis
- Not a replacement for medical professionals
- Always validate predictions clinically

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Developed for educational and research purposes · Last Updated: May 2026*
