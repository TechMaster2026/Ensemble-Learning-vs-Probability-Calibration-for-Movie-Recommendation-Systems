# Ensemble Learning vs Probability Calibration for Movie Recommendation Systems

## Overview
This project explores the effectiveness of **ensemble learning** and **probability calibration** in building movie recommendation systems. The goal is to predict user preferences accurately while providing reliable confidence scores for recommendations. We use the **MovieLens 26M dataset**, containing over 26 million ratings for 45,000 movies.

---

## Dataset
The dataset is sourced from [Kaggle: The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data) and includes:  
- `ratings.csv`: user-movie ratings with columns `userId`, `movieId`, `rating`, `timestamp`.  
- `movies_metadata.csv`: movie metadata including title, genres, budget, revenue, release date, and language.

**Preprocessing steps:**
- Removed missing or invalid ratings.  
- Dropped duplicate user-movie interactions.  
- Categorized ratings:
  - 1 – 2.5 → Not Recommended  
  - 2.6 – 4.0 → Recommended  
  - 4.1 – 5 → Strongly Recommended  
- Selected numeric features (`userId`, `movieId`, `rating`) and applied **StandardScaler**.  
- Applied **Incremental PCA** for dimensionality reduction.  
- Sampled 10% of data for faster computation while retaining essential patterns.

---

## Approach

### Base Models
- **Support Vector Machine (SVM)** – captures non-linear relationships.  
- **Random Forest** – ensemble of decision trees for reduced variance and better generalization.  
- **Naive Bayes** – probabilistic baseline model.

### Stacking Ensemble
- Combines predictions from base models using **Logistic Regression** as the meta-model.  
- Improves overall prediction accuracy and robustness.

### Probability Calibration
- Applied **Platt Scaling (CalibratedClassifierCV)** to base model outputs.  
- Ensures confidence scores are reliable before stacking.  

---

## Implementation
- **Programming Language:** Python 3.12  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn, Dask  
- **Environment:** Google Colab / Jupyter Notebook  

**Pipeline:**
1. Load and preprocess data.  
2. Train base models (SVM, Random Forest, Naive Bayes).  
3. Apply probability calibration to base models.  
4. Train stacking ensemble using Logistic Regression meta-model.  
5. Evaluate performance on test data.

---

## Evaluation Metrics
- **Accuracy** – overall correctness.  
- **Precision** – proportion of correctly predicted positive recommendations.  
- **F1-Score** – balance of precision and recall.  
