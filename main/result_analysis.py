# %%
import pandas as pd
# %%
df_result = pd.read_csv("../result/tuning/xgb_raw_tuning_result.csv")
result = df_result.groupby(['learning_rate', 'max_depth', 'lambda', 'gamma'])['loss'].mean()
result.reset_index(name='loss').query("loss == @result.min()")
# %%
df_result = pd.read_csv("../result/tuning/xgb_imputed_tuning_result.csv")
result = df_result.groupby(['learning_rate', 'max_depth', 'lambda', 'gamma'])['loss'].mean()
result.reset_index(name='loss').query("loss == @result.min()")
# %%
df_result = pd.read_csv("../result/tuning/xgb_fev_tuning_result.csv")
result = df_result.groupby(['learning_rate', 'max_depth', 'lambda', 'gamma'])['loss'].mean()
result.reset_index(name='loss').query("loss == @result.min()")

# %%
df_result = pd.read_csv("../result/tuning/xgb_fev_fvc_tuning_result.csv")
result = df_result.groupby(['learning_rate', 'max_depth', 'lambda', 'gamma'])['loss'].mean()
result.reset_index(name='loss').query("loss == @result.min()")
# %%
