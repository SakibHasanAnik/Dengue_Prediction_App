Here is your fully refined, polished, and **complete, ready-to-use README.md** file tailored for your GitHub repository. It includes clean Markdown syntax, professional structure, and precise data tables.

---

```markdown
# 🦟 Dengue Clinical Outcome Prediction System
---

## 📖 Table of Contents

1. [Overview](#-overview)
2. [Motivation](#-motivation)
3. [Features](#-features)
4. [Dataset](#-dataset)
5. [Methodology](#-methodology)
6. [Model Performance](#-model-performance)
7. [Installation](#-installation)
8. [Usage](#-usage)
9. [Web Application](#-web-application)
10. [Results & Visualizations](#-results--visualizations)
11. [Future Work & Roadmap](#-future-work--roadmap)
12. [Contributing](#-contributing)
13. [License](#-license)
14. [Citation](#-citation)
15. [Contact](#-contact)
16. [Acknowledgments](#-acknowledgments)

---

## 📖 Overview

This project presents a **machine learning-based clinical decision support system** for predicting dengue fever outcomes (Positive/Negative) using patient demographic, clinical, and laboratory features. The system uniquely combines **ensemble learning**, **uncertainty quantification**, and **counterfactual explanations** to provide interpretable, actionable predictions for healthcare professionals.

The model was trained on a real-world dataset of **1,018 patients** from Bangladesh and achieves **99.0% accuracy** with uncertainty-aware predictions.


---

## 🎯 Motivation

Dengue fever affects approximately **390 million people annually** worldwide, with the highest burden in tropical and subtropical regions. In resource-limited settings, diagnostic tools are often unavailable or delayed, making clinical prediction models a valuable alternative for triage and treatment decisions.

### Key Challenges Addressed:
*   **Black-box models:** Most ML models provide predictions without explaining *why*.
*   **No uncertainty quantification:** Point estimates without confidence intervals are clinically risky.
*   **No actionable insights:** Traditional models cannot answer "what if" scenarios for a specific patient.

### Our Solution:
*   **Explainable predictions:** Feature importance paired with explicit counterfactual explanations.
*   **95% confidence intervals:** Uncertainty quantification built directly into each prediction using bootstrap ensembles.
*   **Ambiguous case identification:** Flaps a warning when the confidence interval crosses the 0.5 decision threshold.

---

## ✨ Features

### Core Features
*   **Real-time prediction** of dengue outcome based on non-invasive clinical inputs.
*   **Uncertainty quantification** generating 95% confidence intervals for every patient.
*   **Counterfactual explanations** showing what minimal health metrics would change the diagnosis.
*   **No external advanced lab tests required** (relies primarily on complete blood counts and symptom logs).

### Technical Features
*   **Bootstrap ensemble** with 100 iterations for reliable variance and uncertainty estimation.
*   **Nested cross-validation** to guarantee robust, un-overfitted performance evaluation.
*   **Advanced Feature engineering** (Platelet/WBC ratio, total symptom burden count).
*   **RobustScaler preprocessing** to inherently handle outliers in volatile clinical data.

---

## 📊 Dataset

The dataset consists of **1,018 patient records** collected from multiple clinical locations in Bangladesh.

### Dataset Overview

| Attribute | Details |
|---|---|
| **Total Samples** | 1,018 |
| **Features** | 13 (Reduced to 12 post-preprocessing) |
| **Outcome** | Binary (Positive/Negative) |
| **Class Distribution** | 68.5% Positive, 31.5% Negative |
| **Missing Values** | None |

### Feature Description

| Feature Type | Features | Description |
|---|---|---|
| **Demographic** | Age, Gender, Location | Basic patient profile characteristics |
| **Laboratory** | Platelet Count, WBC | Complete blood count (CBC) parameters |
| **Clinical** | Fever, Duration of Fever | Fever-related tracking data |
| **Symptom** | Headache, Muscle Pain, Rash, Vomiting | Binary symptom indicators |
| **Target** | Outcome | Final Dengue diagnosis (Positive/Negative) |

### Engineered Features
*   `Platelet_WBC_ratio`: Calculated as $\text{Platelet} / (\text{WBC} + 1)$ to capture combined compound hematological effects.
*   `Symptom_count`: Cumulative sum of all 5 binary symptoms to quantify overall symptom burden.

---

## 🔬 Methodology

### 1. Data Preprocessing Pipeline

```

Raw Data ➔ Feature Engineering ➔ Encoding ➔ Scaling ➔ Model Training

```
*   **Encoding:** Gender utilizes Label Encoding ($0=\text{Female}, 1=\text{Male}$). Location uses Frequency Encoding.
*   **Scaling:** Numerical attributes are scaled using `RobustScaler` to minimize the impact of extreme biological outliers.

### 2. Models Implemented

| Model | Parameters | Description |
|---|---|---|
| **Logistic Regression** | `C=1`, `class_weight='balanced'` | Interpretable baseline |
| **Random Forest** | `n_estimators=100`, `max_depth=10` | Ensemble with bagging logic |
| **XGBoost** | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1` | Optimized gradient boosting |
| **Bootstrap Ensemble** | `n_bootstraps=100`, 95% CI | Final uncertainty estimation engine |

### 3. Uncertainty Quantification
We implemented **bootstrap resampling** to evaluate prediction stability across 100 distinct sub-models:

```python
import numpy as np
from sklearn.utils import resample

# Conceptual implementation snippet
preds = []
for i in range(100):
    X_boot, y_boot = resample(X_train, y_train)
    model.fit(X_boot, y_boot)
    preds.append(model.predict_proba(X_test)[:, 1])

mean_pred = np.mean(preds, axis=0)
ci_lower = np.percentile(preds, 2.5, axis=0)
ci_upper = np.percentile(preds, 97.5, axis=0)

```

---

## 📈 Model Performance

### Cross-Validation Results (5-fold)

| Model | AUC (CV) | Std Dev |
| --- | --- | --- |
| Logistic Regression | 0.9947 | ±0.0044 |
| Random Forest | 0.9902 | ±0.0081 |
| XGBoost | 0.9867 | ±0.0099 |

### Test Set Performance

| Model | AUC | Brier Score | Accuracy | CI Width (Mean) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.9940 | 0.0140 | 0.9902 | — |
| Random Forest | 0.9941 | 0.0114 | 0.9902 | — |
| XGBoost | 0.9934 | 0.0105 | 0.9902 | — |
| **Bootstrap Ensemble** | **0.9956** | **0.0118** | **0.9902** | **0.0508** |

### Top Feature Correlations with Outcome

| Feature | Correlation |
| --- | --- |
| Platelet Count | -0.9196 |
| WBC (White Blood Cell) | -0.8763 |
| Symptom Count | +0.5443 |
| Fever | +0.4272 |

---

## 🚀 Installation

### Prerequisites

* Python 3.8 or higher
* `pip` package manager
* `git`

### Setup Instructions

1. **Clone the Repository:**
```bash
git clone [https://github.com/sakibHanik101/dengue-prediction-app.git](https://github.com/sakibHanik101/dengue-prediction-app.git)
cd dengue-prediction-app

```


2. **Create and Activate a Virtual Environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```


3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


4. **Run the Application Locally:**
```bash
streamlit run app.py

```


The application will automatically open at `http://localhost:8501`.

---

## 🖥️ Usage

### Input Schema

The interface accepts standard patient parameters:

* **Age / Gender / Location**
* **Platelet Count** (cells/µL) & **WBC Count** (cells/µL)
* **Symptoms:** Binary checkboxes for Fever, Headache, Muscle Pain, Rash, and Vomiting.

### Example Output Preview

```text
Prediction: Dengue POSITIVE
Probability: 94.7%
Confidence Level: High (CI: [92.1%, 97.3%])

Counterfactual Suggestion: 
If Platelet Count increases to 180,000 cells/µL, prediction changes to NEGATIVE.

```

---

## 📊 Results & Visualizations

The pipeline generates several analytical evaluation metrics automatically saved in the workspace:

* **ROC Curves (`figure1_roc_curves.png`):** Demonstrates model discrimination power across configurations.
* **Calibration Curves (`figure2_calibration_curves.png`):** Confirms that predicted probabilities mirror empirical real-world frequencies.
* **Uncertainty Intervals (`figure3_uncertainty.png`):** Visualizes the 95% CI bands across the cohort to catch borderline clinical cases easily.

---

## 🚧 Future Work & Roadmap

* [x] **Version 1.0 (Current):** Basic bootstrap ensemble, uncertainty metrics validation, and Streamlit deployment.
* [ ] **Version 2.0 (Q4 2026):** Multi-class severity classification (Mild vs. Severe Hemorrhagic Dengue).
* [ ] **Version 3.0 (2027):** Multi-center clinical validation using global data cohorts outside Bangladesh.

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See [LICENSE](https://www.google.com/search?q=LICENSE) for more information.

---

## 📝 Citation

If you use this system or paper elements in your research, please cite it using the following format:

```bibtex
@article{hanik2025dengue,
  title={Explainable Uncertainty-Aware Ensemble Learning for Dengue Clinical Outcome Prediction with Counterfactual Explanations},
  author={Hanik, Sakib},
  journal={Research Square / Preprints},
  year={2025},
  note={Under review}
}

```

---

## 📧 Contact

**Sakib Hanik** - [sakibhanik2003@gmail.com]()

Project Link: [https://github.com/sakibHanik101/dengue-prediction-app](https://github.com/sakibHanik101/dengue-prediction-app)

Hugging Face Profile: [@sakibHanik101](https://huggingface.co/sakibHanik101)

---

## ⭐ Acknowledgments

* Frontline clinical data contributors from Bangladesh.
* The open-source communities behind Streamlit, Scikit-Learn, and DiCE (Diverse Counterfactual Explanations).

```
***

### 💡 Quick Customization Checklist Before Committing:
1. Ensure your root project folder has a `requirements.txt` file listing packages like `streamlit`, `scikit-learn`, `dice-ml`, `xgboost`, etc.
2. If your local figures are named differently, update the filenames under the `## 📊 Results & Visualizations` section. 
3. Populate the `LICENSE` file in your repository root with a standard MIT license text block.

```
