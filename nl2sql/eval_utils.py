# from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from collections import defaultdict
import csv
import numpy
import pandas as pd


def get_metrics_cm_for_one_index(cm, index):
    n_samples = cm.ravel().sum()
    total_obs = cm[index].sum()
    tp = cm[index, index]
    fp = cm[:, index].sum() - cm[index, index]
    fn = cm[index].sum() - cm[index, index]
    tn = n_samples - tp - fp - fn

    recall = tp / float(tp + fn)
    precision = tp / float(tp + fp)
    f1 = (2 * recall * precision) / (recall + precision)
    accuracy = (tp + tn) / n_samples

    if not total_obs == 0:
        recall = recall if not np.isnan(recall) else 0
        f1 = f1 if not np.isnan(f1) else 0
        precision = precision if not np.isnan(precision) else 0
    else:
        recall, f1, precision = 0, 0, 0

    return tp, tn, fp, fn, recall, precision, f1, accuracy, total_obs


# In[81]:

def get_indexwise_metrics(cm):
    report = {}
    for index in range(0, len(cm)):
        tp, tn, fp, fn, recall, precision, f1, accuracy, total_obs = get_metrics_cm_for_one_index(cm, index)
        report[index] = {'tp': tp,
                         'tn': tn,
                         'fp': fp,
                         'fn': fn,
                         'total_obs': total_obs,
                         'recall': recall,
                         'precision': precision,
                         'f1': f1,
                         'accuracy': accuracy}
    return report


# In[86]:

def get_metrics_cm_for_one_slot(metrics_df, start_index, end_index):
    metrics_slot_df = metrics_df.iloc[start_index:end_index]
    slot_report = dict()
    slot_report['total_fn'] = metrics_slot_df['fn'].sum()
    slot_report['total_tn'] = metrics_slot_df['tn'].sum()
    slot_report['total_fp'] = metrics_slot_df['fp'].sum()
    slot_report['total_tp'] = metrics_slot_df['tp'].sum()
    slot_report['total_obs'] = metrics_slot_df['total_obs'].sum()
    slot_report['precision'] = ((metrics_slot_df['precision'] * metrics_slot_df['total_obs'])).sum() / slot_report[
        'total_obs']
    slot_report['recall'] = ((metrics_slot_df['recall'] * metrics_slot_df['total_obs'])).sum() / slot_report[
        'total_obs']
    slot_report['f1'] = ((metrics_slot_df['f1'] * metrics_slot_df['total_obs'])).sum() / slot_report['total_obs']
    slot_report['accuracy'] = ((metrics_slot_df['accuracy'] * metrics_slot_df['total_obs'])).sum() / slot_report[
        'total_obs']

    return slot_report


# In[89]:

def get_slotwise_report(cm, n_words_per_question,
                        n_columns_per_table, n_agg_ops,
                        n_cond_ops):
    indexwise_report = get_indexwise_metrics(cm)
    metrics_df = pd.DataFrame.from_dict(indexwise_report, orient='index')
    slotwise_report = {}

    # start
    slotwise_report['start'] = get_metrics_cm_for_one_slot(metrics_df, 0, 1)

    # aggops
    slotwise_report['aggop'] = get_metrics_cm_for_one_slot(metrics_df, 1, n_agg_ops)

    # selcol
    slotwise_report['selcol'] = get_metrics_cm_for_one_slot(metrics_df, 1 + n_agg_ops,
                                                            (1 + n_agg_ops + n_columns_per_table))

    # condcols
    slotwise_report['condcols'] = get_metrics_cm_for_one_slot(metrics_df, (1 + n_agg_ops + n_columns_per_table),
                                                              (1 + n_agg_ops + 2 * n_columns_per_table))
    # condops
    slotwise_report['condops'] = get_metrics_cm_for_one_slot(metrics_df, (1 + n_agg_ops + 2 * n_columns_per_table),
                                                             (1 + n_agg_ops + 2 * n_columns_per_table + n_cond_ops))

    # condval
    slotwise_report['condval'] = get_metrics_cm_for_one_slot(metrics_df,
                                                             (1 + n_agg_ops + 2 * n_columns_per_table + n_cond_ops),
                                                             (
                                                                         1 + n_agg_ops + 2 * n_columns_per_table + n_cond_ops + n_words_per_question))

    # end
    slotwise_report['end'] = get_metrics_cm_for_one_slot(metrics_df, (
                1 + n_agg_ops + 2 * n_columns_per_table + n_cond_ops + n_words_per_question), (
                                                                     1 + n_agg_ops + 2 * n_columns_per_table + n_cond_ops + n_words_per_question + 1))

    return slotwise_report

# pd.DataFrame.from_dict(slotwise_report,orient='index').to_csv('D://NLP SQL Queries//EvalMetrics.csv')
