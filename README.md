# Development and validation of machine learning prediction models for diagnosing asthma using methacholine challenge test results

**Noeul Kang and Kyung Hyun Lee contributed equally to this work.**

The codes for `Model Development` in Methods of the paper are available.

*** Note that all codes are executable ONLY if your own data exist

## Datasets

- Here, the datasets we used in the paper can not be released for personal information protection.

- Instead, you can identify a sample dataset. Please refer to `sample_dataset.csv`
  - `sample_dataset.csv` shows the examples of the datasets used for modeling (Note that this is not a real subject's dataset)
  - In the "Asthma" column, 0: Normal, 1: Asthma

## Modeling

### 1. XGBoost

- This code covers the gridsearch process and training with the best params for XGBoost

- See `./main/model/gridsearch_xgb.py` and `./main/model/infecrence_xgb.py`

### 2. Logistic Regression, Random Forest, SVM, and ANN

- This code covers the gridsearch process, training with the best params for SVM, RF, ANN, and LR, and inferecing on test data

- See `./main/model/gridsearch_sklearn.py`

```{bash}
cd main
bash ./gridsearch_sklearn.sh
bash ./gridsearch_xgb.sh
bash ./inference_xgb.sh
```
