# %%
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
# %%
## Argparser
parser = argparse.ArgumentParser(description='Study')
parser.add_argument('--feature', type=str)
parser.add_argument('--gpu_id', type=int, default=0)
args = vars(parser.parse_args())
# %%
## Load dataset
DATAPATH = "../data"
SAVEPATH = "../result/test"

df_train = pd.read_csv(os.path.join(DATAPATH, 'train.csv'), encoding='utf-8')
df_test = pd.read_csv(os.path.join(DATAPATH, 'test.csv'), encoding='utf-8')

print("Number of samples = {}".format(len(df_train)))

## Select Feature to Analysis
subs = r"0.05|0.5|2.0|8.0|16.0|32.0"
m_subs = r"imputed|mi"

if args['feature'] == 'raw':
    mbpt_txt = [x for x in df_train.columns if re.search(subs, str(x))]
    mbpt_txt = [x for x in mbpt_txt if not re.search(m_subs, str(x))]
    
    feature = mbpt_txt
    
    param_grid = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'gpu_id': args['gpu_id'],
        'learning_rate': 0.001,
        'max_depth': 2,
        'lambda': 1,
        'gamma': 0.3
    }
    
elif args['feature'] == 'fev':
    mbpt_txt = [x for x in df_train.columns if re.search(subs, str(x))]
    mbpt_txt = [x for x in mbpt_txt if not re.search(m_subs, str(x))]

    get_subs = r"FEV"
    mbpt_txt = [x for x in mbpt_txt if re.search(get_subs, str(x))]

    feature = mbpt_txt
    
    param_grid = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'gpu_id': args['gpu_id'],
        'learning_rate': 0.001,
        'max_depth': 3,
        'lambda': 3,
        'gamma': 0.1
    }
    
elif args['feature'] == 'fev_fvc':
    mbpt_txt = [x for x in df_train.columns if re.search(subs, str(x))]
    mbpt_txt = [x for x in mbpt_txt if not re.search(m_subs, str(x))]

    get_subs = r"FEV|FVC"
    mbpt_txt = [x for x in mbpt_txt if re.search(get_subs, str(x))]

    feature = mbpt_txt
    
    param_grid = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'gpu_id': args['gpu_id'],
        'learning_rate': 0.001,
        'max_depth': 2,
        'lambda': 1,
        'gamma': 0,
    }
else:
    mi_col = [x for x in df_train.columns if re.search(m_subs, str(x))]
    
    feature = mi_col
    
    param_grid = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'gpu_id': args['gpu_id'],
        'learning_rate': 0.001,
        'max_depth': 2,
        'lambda': 1,
        'gamma': 0.2,
    }
    

tuning_result = pd.DataFrame()


train_index, dev_index = train_test_split(
    df_train.index,
    test_size=1/5,
    random_state=1004
)

X_train = df_train.iloc[train_index][feature].values
y_train = df_train.iloc[train_index]['Asthma'].values
d_train = xgb.DMatrix(X_train, label=y_train)

X_dev = df_train.iloc[dev_index][feature].values
y_dev = df_train.iloc[dev_index]['Asthma'].values
d_dev = xgb.DMatrix(X_dev, label=y_dev)

X_test = df_test[feature].values
y_test = df_test['Asthma'].values
d_test = xgb.DMatrix(X_test, label=y_test)

model_xgb = xgb.train(
    param_grid,
    d_train,
    num_boost_round=20000,
    evals=[(d_dev, 'validation')],
    verbose_eval=0,
    early_stopping_rounds=1000
)

test_result = df_test.assign(
    pred_proba = model_xgb.predict(d_test),
    pred_class = lambda x: np.where(x['pred_proba'] > 0.5, 1, 0)
)

test_result.to_csv(
    os.path.join(
        SAVEPATH, 
        f"xgb_{args['feature']}_test_result.csv"
    ), 
    index=False
)
