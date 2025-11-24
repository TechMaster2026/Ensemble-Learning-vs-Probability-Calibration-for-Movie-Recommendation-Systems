## Ensemble Learning vs Probability Calibration for Movie Recommendation Systems

## Overview
This project explores the effectiveness of **ensemble learning** and **probability calibration** in building movie recommendation systems. The goal is to predict user preferences accurately while providing reliable confidence scores for recommendations.  

We use the **MovieLens 26M dataset**, containing over 26 million ratings for 45,000 movies.

---

## Dataset
The dataset is sourced from [Kaggle: The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data) and includes:  

- `ratings.csv`: user-movie ratings with columns `userId`, `movieId`, `rating`, `timestamp`.  
- `movies_metadata.csv`: movie metadata including title, genres, budget, revenue, release date, and language.

### Preprocessing Steps
- Removed missing or invalid ratings.  
- Dropped duplicate user-movie interactions.  
- Categorized ratings:
  - 1 – 2.5 → Not Recommended  
  - 2.6 – 4.0 → Recommended  
  - 4.1 – 5 → Strongly Recommended  
- Selected numeric features (`userId`, `movieId`, `rating`) and applied **StandardScaler**.  
- Applied **Incremental PCA** for dimensionality reduction.  
- Sampled 0.5% of data for faster computation while retaining essential patterns.

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

## Implementation & Experimental Procedure

1. **Load and Preprocess Data**  
   - Clean missing or invalid ratings and remove duplicate user-movie interactions.  
   - Categorize ratings into three classes: Not Recommended, Recommended, Strongly Recommended.  
   - Scale features and apply **Incremental PCA**.  
   - Sample 10% of data to reduce computational load.

2. **Train Individual Base Models**  
   - Models: **SVM**, **Random Forest**, **Naive Bayes**.  
   - Train on the training set and evaluate on the test set to establish baseline performance.

3. **Build Stacking Ensemble**  
   - Use predictions of the base models as meta-features.  
   - Train **Logistic Regression** as the meta-model.  
   - Evaluate on the test set.

4. **Apply Probability Calibration**  
   - Use **Platt Scaling** via `CalibratedClassifierCV` on base models.  
   - Build a **calibrated stacked ensemble** using calibrated probabilities.  

5. **Model Evaluation**  
   - Metrics: **Accuracy**, **Precision**, **F1-Score**.  
   - Probability reliability: **Log Loss** and **Calibration Curves**.  

6. **Comparison and Analysis**  
   - Compare individual base models, stacked ensemble, and calibrated stacked ensemble.  
   - Analyze improvements in predictive accuracy, reliability of probability estimates, and model robustness.

---

## Evaluation Metrics
- **Accuracy** – overall correctness of predictions.  
- **Precision** – proportion of correctly predicted positive recommendations.  
- **F1-Score** – balance of precision and recall.  
- **Probability Calibration** – reliability of predicted probabilities using calibration curves and log loss.

---

## Results
- The stacked ensemble outperforms individual base models in accuracy and F1-Score.  
- Applying Platt Scaling improves the reliability of predicted probabilities without reducing overall accuracy.  
- Calibration curves demonstrate closer alignment between predicted probabilities and observed outcomes.  

---

## Conclusion
- Stacking multiple models improves recommendation accuracy and generalization.  
- Probability calibration ensures that confidence scores are trustworthy for decision-making.  
- This framework is flexible and can be applied to other recommendation datasets.

---

## Environment
- **Programming Language:** Python 3.12  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn, Dask  
- **Execution Environment:** Google Colab / Jupyter Notebook

---

## References
- [MovieLens 26M Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data)  
- Scikit-learn documentation: [Stacking and Calibration](https://scikit-learn.org/stable/modules/ensemble.html)
