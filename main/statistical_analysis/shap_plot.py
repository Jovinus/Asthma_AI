# %%
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)

# %%
## Load dataset
DATAPATH = "../../data"

df_train = pd.read_csv(os.path.join(DATAPATH, 'train.csv'), encoding='utf-8')
df_test = pd.read_csv(os.path.join(DATAPATH, 'test.csv'), encoding='utf-8')

print("Number of samples = {}".format(len(df_train)))

## Select Feature to Analysis
subs = r"0.05|0.5|2.0|8.0|16.0|32.0"
m_subs = r"imputed|mi"

mi_col = [x for x in df_train.columns if re.search(m_subs, str(x))]
feature = mi_col

model = RandomForestClassifier(
    n_jobs=-1, 
    random_state=1004,
    n_estimators=1000,
    class_weight='balanced'
    
)

X_train, y_train = df_train[feature], df_train['Asthma']
X_test, y_test = df_test[feature], df_test['Asthma']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X=X_train, y=y_train)
X_test_scaled = scaler.transform(X=X_test)

model.fit(X=X_train_scaled, y=y_train)

print(model.score(X_test_scaled, y_test))

# %%
shap.initjs()
explainer = shap.TreeExplainer(model=model)
shap_test = explainer.shap_values(X_test_scaled)
# %%

feature_names = [re.sub("imputed_", "", feature) for feature in X_test.columns]
plt.figure(figsize=(10, 10), facecolor='w', dpi=500)
shap.summary_plot(
    shap_values = shap_test[1], 
    features = X_test_scaled, 
    feature_names = feature_names,
    plot_size=(10, 10),
    show=False,
    # color_bar=False
)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
# plt.colorbar()
plt.tight_layout()
plt.savefig("../../result/figure/summary_plot.png")
plt.show()
# %%
