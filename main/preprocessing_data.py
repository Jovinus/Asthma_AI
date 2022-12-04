import os
import pandas as pd
import re

pd.set_option("display.max_columns", None)

DATAPATH = "../data/"


def generate_missing_indicator(df_input: pd.DataFrame) -> pd.DataFrame:

    subs = r"0.05|0.5|2.0|8.0|16.0|32.0"
    mbpt_txt = [x for x in df_input.columns if re.search(subs, str(x))]

    for column in mbpt_txt:
        df_input[f"mi_{column}"] = df_input[column].isnull().astype(int)
        df_input[f"imputed_{column}"] = df_input[column].fillna(value=0)

    df_output = df_input.copy()

    return df_output


def main():
    df_orig = pd.read_csv("../data/raw/asthma_ai_dataset.csv")

    df_orig = df_orig.assign(
        IndexDate=lambda x: x['IndexDate'].astype('datetime64'),
    )

    df_mi = generate_missing_indicator(df_input=df_orig)

    ## Train Test Split
    train_set = df_mi.query('IndexDate.dt.year <= 2018', engine='python')
    test_set = df_mi.query('IndexDate.dt.year > 2018', engine='python')

    ## Save Data
    train_set.to_csv(
        os.path.join(DATAPATH, 'train.csv'),
        index=False,
        encoding="utf-8-sig"
    )
    test_set.to_csv(
        os.path.join(DATAPATH, 'test.csv'),
        index=False,
        encoding="utf-8-sig"
    )


if __name__ == '__main__':
    main()
