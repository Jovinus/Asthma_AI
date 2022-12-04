# %%
import os
import pandas as pd

from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

DATAPATH = "../data/"

# %%





# %%%

def main():
    df_orig = pd.read_csv("../data/raw/asthma_ai_dataset.csv")
    df_orig = df_orig.assign(
        IndexDate = lambda x: x['IndexDate'].astype('datetime64'),
    )

    ## Train Test Split
    train_set, test_set = df_orig.query('IndexDate.dt.year <= 2018', engine='python'), df_orig.query('IndexDate.dt.year > 2018', engine='python')

    ## Save Data
    train_set.to_csv(os.path.join(DATAPATH, 'train.csv'), index=False, encoding="utf-8-sig")
    test_set.to_csv(os.path.join(DATAPATH, 'test.csv'), index=False, encoding="utf-8-sig")
    
# %%
if __name__ == '__main__':
    main()