import numpy as np
import pandas as pd
import os
from scipy.stats import entropy
from sklearn.feature_selection import chi2
import logging
from tqdm import tqdm

def compute_entropy(data):
    """Calculates entropy of the passed `pd.Series`
    """
    p_data = data.value_counts()  # counts occurrence of each value
    entropy_value = entropy(p_data)  # get entropy from counts
    return entropy_value


def gini(data):
    """
    This function calculates gini impurity of a feature.
    input: feature (this needs to be a Pandas series)
    output: gini impurity
    """
    probs = data.value_counts(normalize=True)
    gini = 1 - np.sum(np.square(probs))
    return gini


def information_gain(members, split):
    """
    Measures the reduction in entropy after the split
    :param v: Pandas Series of the members
    :param split:
    :return:
    """
    entropy_before = entropy(members.value_counts(normalize=True))
    split.name = 'split'
    members.name = 'members'
    grouped_distrib = members.groupby(split) \
        .value_counts(normalize=True) \
        .reset_index(name='count') \
        .pivot_table(index='split', columns='members', values='count').fillna(0)
    entropy_after = entropy(grouped_distrib, axis=1)
    entropy_after *= split.value_counts(sort=False, normalize=True)
    return entropy_before - entropy_after.sum()


def comp_feature_information_gain(df, target, descriptive_feature, split_criterion):
    """
    This function calculates information gain for splitting on
    a particular descriptive feature for a given dataset
    and a given impurity criteria.
    Supported split criterion: 'entropy', 'gini'
    """

    if split_criterion == 'entropy':
        target_entropy = compute_entropy(df[target])
    else:
        target_entropy = gini(df[target])

    entropy_list = list()
    weight_list = list()

    for level in df[descriptive_feature].unique():
        df_feature_level = df[df[descriptive_feature] == level]
        if split_criterion == 'entropy':
            entropy_level = compute_entropy(df_feature_level[target])
        else:
            entropy_level = gini(df_feature_level[target])
        entropy_list.append(round(entropy_level, 3))
        weight_level = len(df_feature_level) / len(df)
        weight_list.append(round(weight_level, 3))

    feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))

    return target_entropy - feature_remaining_impurity


# Load data from CSV files
data_dir = '../data_processing/data_features_label_death'
data_static = pd.read_csv(os.path.join(data_dir, 'dataset_static.csv'))
data_dynamic = pd.read_csv(os.path.join(data_dir, 'dataset_dynamic.csv'))
data_labels = pd.read_csv(os.path.join(data_dir, 'dataset_labels.csv'))

# Merge Static and Dynamic tables with labels
full_dataset = data_static.merge(data_dynamic, on='patientid').merge(data_labels, on='patientid')

# Pre-compute all chi_statistics
chi_statistics = chi2(full_dataset, full_dataset['fallecimiento'])[0]

# Print feature scores
with open('./data_analysis/feature_importance/information_metrics/feature_importance.tsv', 'w') as fn:
    print(f'feature\tentropy\tinf_gain_ent\tgini\tinf_gain_gini\tchi-square', file=fn)
    for column in tqdm(full_dataset.columns):
        if column != 'patientid' and column != 'fallecimiento':
            print(f'{column}'
                  f'\t{compute_entropy(full_dataset[column]):.5f}'
                  f'\t{comp_feature_information_gain(full_dataset, "fallecimiento", column, "entropy"):.5f}'
                  f'\t{gini(full_dataset[column]):.5f}'
                  f'\t{comp_feature_information_gain(full_dataset, "fallecimiento", column, "gini"):.5f}'
                  f'\t{chi_statistics[full_dataset.columns.get_loc(column)]:.5f}',
                  file=fn)
