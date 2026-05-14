# Global Weather Classification: A Machine Learning Approach

## 1. Project Overview

**Objective:** The primary goal of this project is to classify global weather conditions into three categories—**Good**, **Moderate**, and **Bad**—using a range of geographic, atmospheric, and air quality features.

**Motivation:** Accurate weather classification is crucial for agriculture, disaster preparedness, logistics, and climate science. This project explores how machine learning models can automate this classification based on readily available meteorological data.

**Key Approach:** The analysis goes beyond simple model fitting. It incorporates a unique ensemble method (Lyapunov Temperature Probit Ensemble) and includes rigorous statistical testing (McNemar, Wilcoxon) and stress tests to evaluate model robustness and calibration.

## 2. Data and Preprocessing

The analysis uses a dataset of global weather observations containing features like temperature, pressure, humidity, wind speed, and various air quality metrics.

### 2.1. Problem Reframing & Feature Engineering
- **Target Variable:** The original `condition_text` (e.g., "Sunny", "Light rain") was transformed into an ordinal categorical target: **0 (Good)**, **1 (Moderate)**, **2 (Bad)**.
- **Feature Selection:** Temperature-related variables (`temperature_fahrenheit`, `feels_like_celsius`) were deliberately excluded to mitigate their role as strong confounders, allowing the model to focus on geographic and atmospheric predictors for better interpretability.
- **New Features:** Several engineered features were created to capture complex relationships:
    - **Statistical Summaries:** Row-wise means, standard deviations, mins, and maxes of all numeric features.
    - **Transformations:** Log, inverse, and square transformations of original features.
    - **Dimensionality Reduction:** PCA and UMAP components were extracted to capture latent structures.
    - **Clustering:** HDBSCAN and Gaussian Mixture Models (GMM) were used to create cluster labels and membership probabilities.
    - **State Modeling:** A Gaussian Hidden Markov Model (HMM) was fit to the PCA-transformed data to infer hidden states.

### 2.2. Data Cleaning & EDA
- **Imputation:** Missing values were imputed using the `SimpleImputer` with the 'most_frequent' strategy.
- **Data Split:** An 80/20 chronological split was used to simulate a realistic time-series scenario.
- **Exploratory Analysis:**
    - `skimpy` summary showed a wide range of numeric features, with air quality variables (`air_quality_Carbon_Monoxide`, `air_quality_PM2.5`) exhibiting high standard deviations, indicating skewed distributions.
    - A correlation heatmap revealed strong positive correlations between `air_quality_PM2.5` and `air_quality_Carbon_Monoxide`.
    - A `distplot` and bar chart showed a highly imbalanced target variable, with most instances labeled as "Good" (0).
    - Boxplots confirmed the presence of significant outliers in many air quality features.

## 3. Modeling and Evaluation

Due to the identified class imbalance, the training data was resampled using **ADASYN (Adaptive Synthetic Sampling)**.

### 3.1. Individual Model Performance
Five models were trained and evaluated on a held-out test set. LightGBM achieved the highest accuracy and F1-score, while CatBoost achieved the best probability calibration (lowest Brier Score and Log Loss).

| Model                 | Accuracy | Precision | Recall  | F1 Score | Log Loss | Brier Score |
|-----------------------|----------|-----------|---------|----------|----------|-------------|
| **LightGBM**          | 0.9038   | 0.8942    | 0.9038  | 0.8947   | 0.4265   | 0.1689      |
| **XGBoost**           | 0.8918   | 0.8811    | 0.8918  | 0.8828   | 0.3643   | 0.1683      |
| **Random Forest**     | 0.8918   | 0.8844    | 0.8918  | 0.8860   | 0.3004   | 0.1643      |
| **CatBoost**          | 0.8878   | 0.8766    | 0.8878  | 0.8796   | 0.2972   | **0.1637**  |
| **Logistic Regression**| 0.8437   | 0.8549    | 0.8437  | 0.8485   | 0.5214   | 0.2457      |

### 3.2. Lyapunov Temperature Probit Ensemble (LTPE)
A custom ensemble was developed to improve probabilistic calibration. It combines predictions from all five base models using:
1.  **Temperature Scaling:** To smooth overconfident probability estimates.
2.  **Probit Transformation:** To map probabilities into a latent Gaussian space.
3.  **Lyapunov-Inspired Stability Weighting:** Models with lower prediction volatility receive higher weights.
4.  **Weighted Probability Fusion.**

**Ensemble Performance:**
- **Accuracy:** 0.898
- **F1 Score:** 0.889
- **Log Loss:** 0.292 (outperformed all individual models)
- **Model Weights:** The ensemble assigned the highest weight to CatBoost (0.345), followed by Logistic Regression (0.198).

### 3.3. Statistical Testing & Robustness
- **McNemar's Test:** No statistically significant difference (p > 0.05) was found between the ensemble's classification decisions and those of the best-performing individual models (LightGBM, CatBoost). This indicates they are statistically equivalent in terms of discrete predictions.
- **Wilcoxon Signed-Rank Test:** Significant differences (p < 0.001) were found in the probability distributions of the ensemble versus the individual models, confirming that the ensemble improves probabilistic calibration.
- **Cliff's Delta (Effect Size):** Confirmed a "large" effect size when comparing the ensemble to models with lower accuracy, but no difference or a "negligible" effect when compared to top performers, aligning with the McNemar results.
- **Quantile Analysis:** The ensemble showed better calibration, with error rates dropping to zero in higher-confidence bins, suggesting more reliable probability estimates.

### 3.4. Stress Testing
To assess robustness, models were tested under four data perturbations:
- **Noise (σ=0.1):** Random Gaussian noise added.
- **Drift (Factor 1.3):** Values scaled.
- **Missingness (15%):** Random features set to zero.
- **Adversarial (ε=0.05):** Small perturbations based on random sign.

**Robustness Ranking (1 = perfect robustness):**
| Model       | Robustness Score |
|-------------|-----------------|
| XGBoost     | 0.869            |
| CatBoost     | 0.863            |
| LightGBM    | 0.838            |
| **Ensemble**| **0.820**        |
| Random Forest| 0.763            |
| LogReg      | 0.262            |

**Conclusion:** Gradient boosting models (XGBoost, CatBoost, LightGBM) are significantly more robust to data perturbations than linear models. While the ensemble improves calibration, it does not necessarily provide a robustness advantage over the best boosting models under stressed conditions.

## 4. Impossibility Testing & Explainability

### 4.1. Impossibility Tests
These tests ensure the model has truly learned a meaningful pattern:
- **Label Randomization:** When the target labels were shuffled, all models' performance dropped to near chance levels (~0.33 accuracy for a 3-class problem), confirming that the models are not memorizing spurious patterns.
- **Feature Projection & PCA Bottleneck:** Performance decreased but remained above chance when features were projected onto random bases or reduced to a single PCA component, indicating a moderate degree of redundancy in the feature space.
- **Noise Dominance:** Increasing noise to high levels destroyed most predictive power, as expected.

### 4.2. Model Explainability (SHAP & Permutation Importance)
- **Global Importance:** Permutation importance for the best-performing models (CatBoost, XGBoost) identified `latitude`, `longitude`, `gmm_` and `hdbscan_` cluster features, and several air quality variables as the most influential predictors.
- **Local Explanations (SHAP):** SHAP summary plots for tree-based models showed that being in a certain cluster or region (e.g., cluster 0 or 2) can push predictions towards 'Good', while higher values of air pollutants like `air_quality_Sulphur_dioxide` and `air_quality_PM2.5` push predictions towards 'Bad'.

## 5. Conclusion

*   **Model Selection is a Trade-off:** LightGBM is the best choice for maximizing discriminative performance (accuracy/F1), while CatBoost is superior for probabilistic calibration (log loss/Brier score). The choice depends on the application's needs.
*   **Ensemble for Calibration:** The proposed Lyapunov Temperature Probit Ensemble significantly improves probability calibration without a statistically significant gain in classification accuracy. Its primary value lies in creating more reliable uncertainty estimates.
*   **Robustness is Key:** Gradient boosting models are far more resilient to data distribution shifts than simpler models. XGBoost demonstrated the highest overall robustness.
*   **Model Reliability:** Impossibility tests confirm that the models are learning genuine relationships, not memorizing noise.

## 6. How to Run

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-project-directory>
    ```
2.  **Set up Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Run the Analysis:**
    The main analysis is contained in the Jupyter Notebook `GlobalForecasting.ipynb`. Run all cells to execute the data loading, preprocessing, feature engineering, model training, and evaluation steps.
4.  **View Results:**
    - Model metrics, plots (confusion matrices), and SHAP analyses will be displayed within the notebook.
    - All runs and metrics are automatically logged to **MLflow**. To view the MLflow UI, run `mlflow ui` in the terminal and open the provided local URL (e.g., `http://127.0.0.1:5000`). This provides a comprehensive record of all experiments, parameters, and metrics.
