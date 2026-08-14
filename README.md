# machine-learning-project-
# Chronic Kidney Disease (CKD) Prediction

Course project — CBIO313: Data Mining & Machine Learning.
An end-to-end machine learning pipeline that predicts whether a patient has
Chronic Kidney Disease from routine clinical and laboratory measurements.

## Project description

CKD often has no symptoms until kidney function is already significantly
reduced. This project builds a classifier from 24 routine clinical/lab
features (blood pressure, blood glucose, hemoglobin, serum creatinine, etc.)
to flag patients likely to have CKD, so follow-up testing can happen earlier.
The full workflow — data cleaning, EDA, feature engineering/selection, model
tuning, evaluation, and deployment — is documented step by step in
`notebook.ipynb`.

## Dataset source

UCI Machine Learning Repository — Chronic Kidney Disease dataset
(400 patients, 25 raw columns, collected at a hospital in Tamil Nadu, India).
`https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease`
The raw (uncleaned) version used here is in `data/kidney_disease.csv` and
contains missing values, stray `?` placeholders, tab characters, and an
inconsistent target label — all handled explicitly in the notebook.

## Machine learning algorithms used

Three algorithms were trained and tuned with 5-fold `GridSearchCV`, then
compared on a held-out test set:

| Model | Tuned hyperparameters |
|---|---|
| Logistic Regression | `C` |
| Random Forest | `n_estimators`, `max_depth` |
| SVM | `C`, `kernel` |

The best model by test-set F1-score is saved to `models/best_model.pkl` and
used by the deployed app.

## How to run the project

**1. Run the analysis notebook**

```bash
jupyter notebook notebook.ipynb
```

Run all cells top to bottom. This reproduces cleaning, EDA, feature
engineering/selection, model training/tuning, evaluation, and re-saves the
model artifacts into `models/`.

**2. Deploy publicly (for the project's web-app requirement)**

Push this repo to GitHub (public), then deploy for free on
[Streamlit Community Cloud](https://share.streamlit.io): New app → select
this repo → main file `app.py` → Deploy. Add the resulting public URL here:


## References

* UCI Machine Learning Repository — Chronic Kidney Disease Data Set.
* Scikit-learn documentation — `https://scikit-learn.org`
* Streamlit documentation — `https://docs.streamlit.io`

The app link: https://ckdproject.streamlit.app/
video presentation: https://drive.google.com/file/d/1QLRNjTFML5v3WM9egyiUkE-c6fkJr4pK/view?usp=sharing

