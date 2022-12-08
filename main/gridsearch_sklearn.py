import argparse
import re
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

pd.set_option("display.max_columns", None)

## Argparser
parser = argparse.ArgumentParser(description='Study')
parser.add_argument('--feature', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--n_jobs', type=int, default=1)
args = vars(parser.parse_args())

## Load dataset
DATAPATH = "../data"
SAVEPATH_TUNING = "../result/tuning"
SAVEPATH_TEST = "../result/test"

df_train = pd.read_csv(os.path.join(DATAPATH, 'train.csv'), encoding='utf-8')
df_test = pd.read_csv(os.path.join(DATAPATH, 'test.csv'), encoding='utf-8')

print("Number of samples = {}".format(len(df_train)))

## Check missings
subs = r"0.05|0.5|2.0|8.0|16.0|32.0"
mbpt_txt = [x for x in df_train.columns if re.search(subs, str(x))]

subs = r"imputed|mi"
mi_col = [x for x in mbpt_txt if re.search(subs, str(x))]

if args['feature'] == 'raw':
    feature = mbpt_txt
else:
    feature = mi_col

## Hyperparameter
if args['model'] == 'logistic':
    param_grids = [
        {
            'penalty':['l2'],
            'C':[0.01, 0.1, 1, 10],
        },
        
        {
            'penalty':['none'],
            'C':[1],
        },
    ]
    
    model = LogisticRegression(max_iter=1000, n_jobs=1, random_state=1004)
    
elif args['model'] == 'rf':
    param_grids = {
        'n_estimators':[100, 500, 1000],
        'class_weight':['balanced', 'balanced_subsample']
    }
    model = RandomForestClassifier(n_jobs=-1, random_state=1004)
    
elif args['model'] == 'svm':
    param_grids = [
        {
            'kernel':['linear'],
            'C':[0.01, 0.1, 1, 10, 100],
        },
        {
            'kernel':['rbf'],
            'C':[0.01, 0.1, 1, 10, 100],
            'gamma':['scale', 'auto']
        },
        {
            'kernel':['poly'],
            'degree':[2, 3, 4],
            'C':[0.01, 0.1, 1, 10, 100],
            'gamma':['scale', 'auto']
        },
    ]
    model = SVC(random_state=1004, probability=True)
    
elif args['model'] == 'ann':
    param_grids = {
        'hidden_layer_sizes': [
            (300, 150 , 75),
            (200, 100 , 50),
            (100, 50 , 25),
        ],
        'learning_rate': ['constant', 'adaptive'],
    }
    model = MLPClassifier(
        random_state=1004, 
        early_stopping=True, 
        validation_fraction=1/4,
        max_iter=1000,
        n_iter_no_change=50,
    )

gridsearch = GridSearchCV(
    estimator=model,
    param_grid=param_grids,
    scoring='neg_log_loss',
    cv=5,
    n_jobs=args['n_jobs'], 
    refit=True,
)

X_train, y_train = df_train[feature], df_train['Asthma']
X_test, y_test = df_test[feature], df_test['Asthma']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X=X_train, y=y_train)
X_test_scaled = scaler.transform(X=X_test)

gridsearch.fit(X=X_train_scaled, y=y_train)

test_result = df_test.assign(
    pred_class = gridsearch.predict(X=X_test_scaled),
    pred_proba = gridsearch.predict_proba(X=X_test_scaled)[:, 1],
)

tuning_result = pd.DataFrame(gridsearch.cv_results_)

tuning_result.to_csv(
    os.path.join(
        SAVEPATH_TUNING, 
        f"{args['model']}_{args['feature']}_tuning_result.csv"
    ), 
    index=False
)

test_result.to_csv(
    os.path.join(
        SAVEPATH_TEST, 
        f"{args['model']}_{args['feature']}_test_result.csv"
    ), 
    index=False
)
