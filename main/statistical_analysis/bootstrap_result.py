# %%
import os
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, \
    average_precision_score, precision_score, recall_score
from sklearn.utils import resample

pd.set_option("display.max_columns", None)

# %%
def calculate_bootstrap_metric(
    df_log:pd.DataFrame, 
    target:str, 
    predict:str, 
    n_bootstrap:int,
    model:str,
) -> None:

    
    acc = []
    f1 = []
    auroc = []
    auprc = []
    precision = []
    recall = []
    specificity = []
    

    for i in range(n_bootstrap):
        sampled_df = resample(
            df_log, 
            replace=True, 
            n_samples=300, 
            random_state=i
        )
        
        calc_acc = accuracy_score(
            sampled_df[target], 
            sampled_df[predict], 
        )
        
        calc_precision = precision_score(
            sampled_df[target], 
            sampled_df[predict], 
        )
        
        calc_recall = recall_score(
            sampled_df[target], 
            sampled_df[predict], 
        )
        
        calc_specificity = recall_score(
            sampled_df[target], 
            sampled_df[predict], 
            pos_label=0
        )
        
        calc_f1 = f1_score(
            sampled_df[target], 
            sampled_df[predict], 
            average="binary"
        )
        
        if model == "conventional":
            calc_auroc = roc_auc_score(
                sampled_df[target], 
                sampled_df["MBPT_result"], 
            )
            
            calc_auprc = average_precision_score(
                sampled_df[target], 
                sampled_df["MBPT_result"], 
            )
        else:
            calc_auroc = roc_auc_score(
                sampled_df[target], 
                sampled_df['pred_proba'], 
            )
            
            calc_auprc = average_precision_score(
                sampled_df[target], 
                sampled_df['pred_proba'], 
            )
        
        acc.append(calc_acc)
        precision.append(calc_precision)
        recall.append(calc_recall)
        specificity.append(calc_specificity)
        f1.append(calc_f1)
        auroc.append(calc_auroc)
        auprc.append(calc_auprc)
        
    print(f"Accuracy = {np.mean(acc):.3f} +- {np.std(acc):.3f}")
    print(f"Precision = {np.mean(precision):.3f} +- {np.std(precision):.3f}")
    print(f"Recall(Sensitivity) = {np.mean(recall):.3f} +- {np.std(recall):.3f}")
    print(f"Specificity = {np.mean(specificity):.3f} +- {np.std(specificity):.3f}")
    print(f"F1 Score = {np.mean(f1):.3f} +- {np.std(f1):.3f}")
    print(f"AUROC = {np.mean(auroc):.3f} +- {np.std(auroc):.3f}")
    print(f"AUPRC = {np.mean(auprc):.3f} +- {np.std(auprc):.3f}\n")
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": recall,
        "f1": f1, 
        "auroc": auroc, 
        "auprc": auprc, 
        "model": model,
    }
    
    metrics_df = pd.DataFrame(metrics)
        
    return metrics_df

# %%
if __name__ == "__main__":
    DATAPATH = "../../result/test"
    SAVEPATH = "../../result/bootstrapped"
    
    total_result = []
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "xgb_raw_test_result.csv"))
    
    print("Conventional")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="MBPT_result",
        n_bootstrap=100,
        model='conventional',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "xgb_raw_results.csv"), index=False)
    
    total_result.append(result)
    
    print("XGBoost Raw")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='xgb_raw',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "xgb_raw_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "xgb_imputed_test_result.csv"))
    
    print("XGBoost Imputed")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='xgb_imputed',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "xgb_imputed_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "xgb_fev_test_result.csv"))
    
    print("XGBoost FEV")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='xgb_fev',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "xgb_fev_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "xgb_fev_fvc_test_result.csv"))
    
    print("XGBoost FEV_FVC")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='xgb_fev_fvc',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "xgb_fev_fvc_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "ann_imputed_test_result.csv"))
    print("ANN Imputed")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='ann_imputed',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "ann_imputed_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "ann_fev_test_result.csv"))
    print("ANN FEV")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='ann_fev',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "ann_fev_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "ann_fev_fvc_test_result.csv"))
    print("ANN FEV_FVC")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='ann_fev_fvc',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "ann_fev_fvc_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "svm_imputed_test_result.csv"))
    print("SVM Imputed")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='svm_imputed',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "svm_imputed_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "svm_fev_test_result.csv"))
    print("SVM FEV")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='svm_fev',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "svm_fev_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "svm_fev_fvc_test_result.csv"))
    print("SVM FEV_FVC")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='svm_fev_fvc',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "svm_fev_fvc_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "rf_imputed_test_result.csv"))
    print("Random Forest Imputed")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='rf_imputed',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "rf_imputed_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "rf_fev_test_result.csv"))
    print("Random Forest FEV")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='rf_fev',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "rf_fev_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "rf_fev_fvc_test_result.csv"))
    print("Random Forest FEV_FVC")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='rf_fev_fvc',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "rf_fev_fvc_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "logistic_imputed_test_result.csv"))
    print("Logistic Imputed")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='logistic_imputed',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "logistic_imputed_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "logistic_fev_test_result.csv"))
    print("Logistic FEV")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='logistic_fev',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "logistic_fev_results.csv"), index=False)
    total_result.append(result)
    
    df_orig = pd.read_csv(os.path.join(DATAPATH, "logistic_fev_fvc_test_result.csv"))
    print("Logistic FEV_FVC")
    result = calculate_bootstrap_metric(
        df_log=df_orig,
        target="Asthma",
        predict="pred_class",
        n_bootstrap=100,
        model='logistic_fev_fvc',
    )
    
    result.to_csv(os.path.join(SAVEPATH, "logistic_fev_fvc_results.csv"), index=False)
    total_result.append(result)
    
    total_result_df = pd.concat(total_result, ignore_index=True)
    total_result_df.to_csv(os.path.join(SAVEPATH, "total_results.csv"), index=False)
# %%
