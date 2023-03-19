import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tableone import TableOne

from glob import glob
from sklearn.metrics import roc_curve, precision_recall_curve, PrecisionRecallDisplay
from IPython.display import display

pd.set_option("display.max_columns", None)


def calc_interp_curve(dataframe, y, score):

    fpr_base = np.arange(0, 1, step=0.001)
    ## Conventional
    fpr, tpr, thresholds = roc_curve(
        y_true=dataframe[y],
        y_score=dataframe[score],
        pos_label=1,
    )

    tpr_interp = np.interp(fpr_base, fpr, tpr)
    tpr_interp[0] = 0
    tpr_interp[-1] = 1
    threshold_interp = np.interp(fpr_base, fpr, thresholds)

    return fpr_base, tpr_interp, threshold_interp


def get_roc_curve(file_list):

    dataframe = pd.read_csv(file_list[0])

    ## Conventional
    fpr_base, tpr_interp, thresholds_interp = calc_interp_curve(
        dataframe=dataframe,
        y="Asthma",
        score="MBPT_result"
    )

    conventional_result = {
        'fpr': fpr_base,
        'tpr': tpr_interp,
        'thresholds': thresholds_interp,
        'model_auroc': "Conventional (0.856, 0.852-0.861)",
    }
    conventional_result = pd.DataFrame(conventional_result)

    ## SVM
    fpr_base, tpr_interp, thresholds_interp = calc_interp_curve(
        dataframe=dataframe,
        y="Asthma",
        score="pred_proba"
    )

    svm_result = {
        'fpr': fpr_base,
        'tpr': tpr_interp,
        'thresholds': thresholds_interp,
        'model_auroc': "SVM (0.935, 0.932-0.938)",
    }
    svm_result = pd.DataFrame(svm_result)

    ## Logistic Regression
    dataframe = pd.read_csv(file_list[1])
    fpr_base, tpr_interp, thresholds_interp = calc_interp_curve(
        dataframe=dataframe,
        y="Asthma",
        score="pred_proba"
    )

    lr_result = {
        'fpr': fpr_base,
        'tpr': tpr_interp,
        'thresholds': thresholds_interp,
        'model_auroc': "LR (0.933, 0.9330-0.936)",
    }
    lr_result = pd.DataFrame(lr_result)

    ## ANN
    dataframe = pd.read_csv(file_list[2])
    fpr_base, tpr_interp, thresholds_interp = calc_interp_curve(
        dataframe=dataframe,
        y="Asthma",
        score="pred_proba"
    )

    ann_result = {
        'fpr': fpr_base,
        'tpr': tpr_interp,
        'thresholds': thresholds_interp,
        'model_auroc': "ANN (0.942, 0.940-0.945)",
    }
    ann_result = pd.DataFrame(ann_result)

    ## XGB
    dataframe = pd.read_csv(file_list[3])
    fpr_base, tpr_interp, thresholds_interp = calc_interp_curve(
        dataframe=dataframe,
        y="Asthma",
        score="pred_proba"
    )

    xgb_result = {
        'fpr': fpr_base,
        'tpr': tpr_interp,
        'thresholds': thresholds_interp,
        'model_auroc': "XGB (0.941, 0.938-0.943)",
    }
    xgb_result = pd.DataFrame(xgb_result)

    ## ANN
    dataframe = pd.read_csv(file_list[4])
    fpr_base, tpr_interp, thresholds_interp = calc_interp_curve(
        dataframe=dataframe,
        y="Asthma",
        score="pred_proba"
    )

    rf_result = {
        'fpr': fpr_base,
        'tpr': tpr_interp,
        'thresholds': thresholds_interp,
        'model_auroc': "RF (0.950, 0.948-0.952)",
    }
    rf_result = pd.DataFrame(rf_result)

    result = pd.concat(
        [
            conventional_result, svm_result,
            lr_result, ann_result,
            xgb_result,
            rf_result
        ],
        ignore_index=True
    )

    return result


def calc_interp_prc(dataframe, y, score):

    recall_base = np.arange(0, 1, step=0.001)

    ## Conventional
    precision, recall, thresholds = precision_recall_curve(
        y_true=dataframe[y],
        probas_pred=dataframe[score],
        pos_label=1,
    )

    precision = np.flip(precision)
    recall = np.flip(recall)

    precision_interp = np.interp(recall_base, recall, precision)
    precision_interp[-1] = 0
    precision_interp[0] = 1
    # threshold_interp = np.interp(recall_base, recall, thresholds)

    return recall_base, precision_interp  # , threshold_interp


def get_prc_curve(file_list):

    dataframe = pd.read_csv(file_list[0])

    ## Conventional
    recall_base, precision_interp = calc_interp_prc(
        dataframe=dataframe,
        y="Asthma",
        score="MBPT_result"
    )

    conventional_result = {
        'recall': recall_base,
        'precision': precision_interp,
        # 'thresholds':thresholds_interp,
        'model_auprc': "Conventional (0.759, 0.751-0.766)",
    }
    conventional_result = pd.DataFrame(conventional_result)

    ## SVM
    recall_base, precision_interp = calc_interp_prc(
        dataframe=dataframe,
        y="Asthma",
        score="pred_proba"
    )

    svm_result = {
        'recall': recall_base,
        'precision': precision_interp,
        # 'thresholds':thresholds_interp,
        'model_auprc': "SVM (0.868, 0.860-0.875)",
    }
    svm_result = pd.DataFrame(svm_result)

    ## Logistic Regression
    dataframe = pd.read_csv(file_list[1])
    recall_base, precision_interp = calc_interp_prc(
        dataframe=dataframe,
        y="Asthma",
        score="pred_proba"
    )

    lr_result = {
        'recall': recall_base,
        'precision': precision_interp,
        # 'thresholds':thresholds_interp,
        'model_auprc': "LR (0.900, 0.885-0.895)",
    }
    lr_result = pd.DataFrame(lr_result)

    ## ANN
    dataframe = pd.read_csv(file_list[2])
    recall_base, precision_interp = calc_interp_prc(
        dataframe=dataframe,
        y="Asthma",
        score="pred_proba"
    )

    ann_result = {
        'recall': recall_base,
        'precision': precision_interp,
        # 'thresholds':thresholds_interp,
        'model_auprc': "ANN (0.890, 0.885-0.896)",
    }
    ann_result = pd.DataFrame(ann_result)

    ## XGB
    dataframe = pd.read_csv(file_list[3])
    recall_base, precision_interp = calc_interp_prc(
        dataframe=dataframe,
        y="Asthma",
        score="pred_proba"
    )

    xgb_result = {
        'recall': recall_base,
        'precision': precision_interp,
        # 'thresholds':thresholds_interp,
        'model_auprc': "XGB (0.896, 0.891-0.901)",
    }
    xgb_result = pd.DataFrame(xgb_result)

    ## ANN
    dataframe = pd.read_csv(file_list[4])
    recall_base, precision_interp = calc_interp_prc(
        dataframe=dataframe,
        y="Asthma",
        score="pred_proba"
    )

    rf_result = {
        'recall': recall_base,
        'precision': precision_interp,
        # 'thresholds':thresholds_interp,
        'model_auprc': "RF (0.909, 0.905-0.914)",
    }
    rf_result = pd.DataFrame(rf_result)

    result = pd.concat(
        [
            conventional_result, svm_result,
            lr_result, ann_result,
            xgb_result, rf_result
        ],
        ignore_index=True
    )

    return result


def plot_roc_curve_figure():

    file_list = glob(r"../../result/test/*impu*")

    result = get_roc_curve(file_list=file_list)
    hue_order = [
        "Conventional (0.856, 0.852-0.861)",
        "RF (0.950, 0.948-0.952)",
        "ANN (0.942, 0.940-0.945)",
        "XGB (0.941, 0.938-0.943)",
        "SVM (0.935, 0.932-0.938)",
        "LR (0.933, 0.9330-0.936)",
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
    sns.lineplot(
        data=result,
        x='fpr',
        y='tpr',
        hue='model_auroc',
        hue_order=hue_order,
        ax=ax,
        linewidth=2.5
    )

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("1 - Specificity", fontsize=20)
    plt.ylabel("Sensitivity", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("../../result/figure/auroc_plot.png", dpi=500)
    plt.show()

    return None


def plot_prc_curve_figure():

    file_list = glob(r"../../result/test/*impu*")

    result = get_prc_curve(file_list=file_list)
    hue_order = [
        "Conventional (0.759, 0.751-0.766)",
        "RF (0.909, 0.905-0.914)",
        "ANN (0.890, 0.885-0.896)",
        "XGB (0.896, 0.891-0.901)",
        "SVM (0.868, 0.860-0.875)",
        "LR (0.900, 0.885-0.895)",
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
    sns.lineplot(
        data=result,
        x='recall',
        y='precision',
        hue='model_auprc',
        hue_order=hue_order,
        ax=ax,
        linewidth=2.5
    )

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("../../result/figure/auprc_plot.png", dpi=500)
    plt.show()

    return None


if __name__ == "__main__":
    df_orig = pd.read_csv("../../result/bootstrapped/total_results.csv")

    df_orig.groupby('model')['auroc'].mean()

    upper = df_orig.groupby('model')['auroc'].mean(
    ) + (df_orig.groupby('model')['auroc'].std() * 1.96 / 10)
    lower = df_orig.groupby('model')['auroc'].mean(
    ) - (df_orig.groupby('model')['auroc'].std() * 1.96 / 10)

    print(upper)
    print(lower)

    df_orig.groupby('model')['auprc'].mean()

    upper = df_orig.groupby('model')['auprc'].mean(
    ) + (df_orig.groupby('model')['auprc'].std() * 1.96 / 10)
    lower = df_orig.groupby('model')['auprc'].mean(
    ) - (df_orig.groupby('model')['auprc'].std() * 1.96 / 10)

    print(upper)
    print(lower)

    plot_roc_curve_figure()
    plot_prc_curve_figure()

    file_list = glob(r"../../result/test/*impu*")

    dataframe = pd.read_csv(file_list[4])

    recall_base = np.arange(0, 1, step=0.01)

    ## Conventional
    precision, recall, thresholds = precision_recall_curve(
        y_true=dataframe['Asthma'],
        probas_pred=dataframe['pred_proba'],
        pos_label=1,
    )

    precision = np.flip(precision)
    recall = np.flip(recall)

    precision_interp = np.interp(recall_base, recall, precision)

    df_orig = pd.read_csv("../../result/bootstrapped/total_results.csv")

    df_to_stat = (
        df_orig
        .query("model.isin(['conventional', 'rf_imputed'])")
        .reset_index(drop=True)
    )

    my_table = TableOne(
        data=df_to_stat,
        columns=[
            'auroc', 'auprc', 'recall',
            'specificity', 'ppv', 'npv', 
            'accuracy'
        ],
        groupby=['model'],
        pval=True,
        pval_test_name=True,
        decimals=3
    )

    print(my_table)
