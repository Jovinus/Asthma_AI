import argparse
import re
import os
import numpy as np
import pandas as pd
import xgboost as xgb

from IPython.display import display
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
from tqdm import tqdm

pd.set_option("display.max_columns", None)

## Argparser
parser = argparse.ArgumentParser(description='Study')
parser.add_argument('--feature', type=str)
parser.add_argument('--gpu_id', type=int, default=0)
args = vars(parser.parse_args())

## Load dataset
DATAPATH = "../data"
SAVEPATH = "../result/tuning"

df_orig = pd.read_csv(os.path.join(DATAPATH, 'train.csv'), encoding='utf-8')

print("Number of samples = {}".format(len(df_orig)))

## Check missings
subs = r"0.05|0.5|2.0|8.0|16.0|32.0"
mbpt_txt = [x for x in df_orig.columns if re.search(subs, str(x))]

subs = r"imputed|mi"
mi_col = [x for x in mbpt_txt if re.search(subs, str(x))]

if args['feature'] == 'raw':
    feature = mbpt_txt
else:
    feature = mi_col

## Hyperparameter
param_grids = {
    'objective': ['binary:logistic'],
    'eval_metric': ['logloss'],
    'tree_method': ['gpu_hist'],
    'gpu_id': [args['gpu_id']],
    'learning_rate': [0.01, 0.001],
    'max_depth': [1, 2, 3, 4, 5],
    'lambda': [1, 2, 3],
    'gamma': [0, 0.1, 0.2, 0.3]
}

tuning_result = pd.DataFrame()

for param_grid in tqdm(ParameterGrid(param_grids)):

    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4)

    loss = []

    for train_dev_index, eval_index in skf.split(df_orig, df_orig['Asthma']):

        train_index, dev_index = train_test_split(
            train_dev_index,
            test_size=1/4,
            random_state=1004
        )

        X_train = df_orig.iloc[train_index][feature].values
        y_train = df_orig.iloc[train_index]['Asthma'].values
        d_train = xgb.DMatrix(X_train, label=y_train)

        X_dev = df_orig.iloc[dev_index][feature].values
        y_dev = df_orig.iloc[dev_index]['Asthma'].values
        d_dev = xgb.DMatrix(X_dev, label=y_dev)

        X_eval = df_orig.iloc[eval_index][feature].values
        y_eval = df_orig.iloc[eval_index]['Asthma'].values
        d_eval = xgb.DMatrix(X_eval, label=y_eval)

        model_xgb = xgb.train(
            param_grid,
            d_train,
            num_boost_round=20000,
            evals=[(d_dev, 'validation')],
            verbose_eval=0,
            early_stopping_rounds=1000
        )

        y_pred_proba_val = model_xgb.predict(d_eval)

        loss.append(
            log_loss(y_eval, y_pred_proba_val)
        )

    tmp_result = pd.DataFrame(
        {
            "learning_rate": np.full(20, float(param_grid["learning_rate"])).tolist(),
            "max_depth": np.full(20, float(param_grid["max_depth"])).tolist(),
            "lambda": np.full(20, float(param_grid["lambda"])).tolist(),
            "gamma": np.full(20, float(param_grid["gamma"])).tolist(),
            "cv": np.arange(0, 20).tolist(),
            "loss": loss
        }
    )

    tuning_result = pd.concat(
        (tuning_result, tmp_result), axis=0).reset_index(drop=True)

tuning_result.to_csv(
    os.path.join(
        SAVEPATH, 
        f"xgb_{args['feature']}_tuning_result.csv"
    ), 
    index=False
)
