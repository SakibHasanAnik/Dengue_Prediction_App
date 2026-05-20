# 🦟 Dengue Clinical Outcome Prediction System

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue)](https://huggingface.co/spaces/sakibHanik101/Dengue_Prediction_App)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Web Application](#web-application)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## 📖 Overview

This project presents a **machine learning-based clinical decision support system** for predicting dengue fever outcomes (Positive/Negative) using patient demographic, clinical, and laboratory features. The system combines **ensemble learning**, **uncertainty quantification**, and **counterfactual explanations** to provide interpretable, actionable predictions for healthcare professionals.

The model was trained on a real-world dataset of **1,018 patients** from Bangladesh and achieves **98.5% accuracy** with uncertainty-aware predictions.

---

## 🎯 Motivation

Dengue fever affects approximately **390 million people annually** worldwide, with the highest burden in tropical and subtropical regions. In resource-limited settings, diagnostic tools are often unavailable or delayed, making clinical prediction models a valuable alternative for triage and treatment decisions.

**Key challenges addressed:**
- ❌ Black-box models lack interpretability
- ❌ No uncertainty quantification in predictions
- ❌ No actionable "what-if" explanations

**Our solution:**
- ✅ Explainable predictions with feature importance
- ✅ 95% confidence intervals for uncertainty
- ✅ Counterfactual explanations for clinical actionability

---

## ✨ Features

- **Real-time prediction** of dengue outcome based on clinical inputs
- **Uncertainty quantification** with 95% confidence intervals
- **Counterfactual explanations** showing what would change the diagnosis
- **Interpretable model** with feature importance analysis
- **Web-based interface** for easy clinical use
- **No external lab tests required** (uses complete blood count and symptoms)

---

## 📊 Dataset

The dataset consists of **1,018 patient records** collected from multiple locations in Bangladesh.

| Feature Type | Features | Description |
|--------------|----------|-------------|
| Demographic | Age, Gender, Location | Patient characteristics |
| Laboratory | Platelet Count, WBC | Complete blood count parameters |
| Clinical | Fever, Duration of Fever | Fever-related features |
| Symptom | Headache, Muscle Pain, Rash, Vomiting | Binary symptom indicators |
| Outcome | Positive/Negative | Dengue diagnosis (target variable) |

**Class distribution:** 68.5% Positive, 31.5% Negative

---

## 🔬 Methodology

### Data Preprocessing
- **Feature engineering:** Platelet/WBC ratio, symptom count
- **Encoding:** Label encoding for gender, frequency encoding for location
- **Scaling:** RobustScaler for handling outliers
- **Feature selection:** 12 features after preprocessing

### Models Implemented
| Model | Description |
|-------|-------------|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble learning with bagging |
| XGBoost | Gradient boosting with regularization |
| Bootstrap Ensemble | 100 iterations with 95% CI for uncertainty |

### Uncertainty Quantification
- **Bootstrap resampling** (n=100) to estimate prediction distributions
- **95% confidence intervals** for each prediction
- **Ambiguous case identification** where CI crosses 0.5 threshold

### Counterfactual Explanations
- Generated using **DiCE** (Diverse Counterfactual Explanations)
- Answers: *"What minimal feature changes would flip the diagnosis?"*
- Provides actionable clinical insights

---

## 📈 Model Performance

| Model | AUC | Brier Score | Accuracy | CI Width (mean) |
|-------|-----|-------------|----------|-----------------|
| Logistic Regression | 0.9940 | 0.0140 | 0.9902 | - |
| Random Forest | 0.9941 | 0.0114 | 0.9902 | - |
| XGBoost | 0.9934 | 0.0105 | 0.9902 | - |
| **Bootstrap Ensemble** | **0.9956** | **0.0118** | **0.9902** | **0.0508** |

**Key findings:**
- All models achieve >99% AUC due to strong clinical signals
- Bootstrap Ensemble provides uncertainty estimates without sacrificing performance
- **0.98%** of test cases identified as ambiguous (CI crosses 0.5)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dengue-prediction-app.git
cd dengue-prediction-app
