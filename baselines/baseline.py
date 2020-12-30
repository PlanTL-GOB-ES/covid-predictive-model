import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import make_scorer, get_scorer
from tqdm import tqdm
import pprint
import random
import json
from collections import defaultdict

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
SELECT_METRIC = 'test_f1'
METRIC = SELECT_METRIC.split('_')[1]
HOSPITAL = 'h12o'
DATASET_BASE_NAME = 'death_v1_base'
DATASET_NAME = 'death_v1_with_inputation_with_missing_with_reference_values'
pp = pprint.PrettyPrinter(indent=4)

if SELECT_METRIC == 'test_f1':
    metric = metrics.f1_score
    metric_params = {'average': 'macro'}
else:
    metric = metrics.accuracy_score
    metric_params = {}


def compute_specificity(corrects_all, predicted_all):
    tn, fp, fn, tp = metrics.confusion_matrix(corrects_all, predicted_all).ravel()
    specificity = tn / (tn + fp)
    return specificity


def get_hits(corrects_all, predicted_all):
    return sum([1 for x in zip(corrects_all, predicted_all) if x[0] == x[1]])


def get_f1(corrects_all, predicted_all):
    return metrics.f1_score(corrects_all, predicted_all, average='macro')


def get_totals(corrects_all, _):
    return len(corrects_all)


def get_fold_results(corrects_all, predicted_all, predicted_all_probs, data):
    data = data.reset_index()
    group = data.groupby('patientid', sort=False, as_index=False)
    patients = data['patientid'].to_numpy()
    corrects = defaultdict(list)
    predicted = defaultdict(list)
    predicted_probs = defaultdict(list)
    patient_ids = defaultdict(list)
    for i in range(group.size().max()):
        idxs = group.nth(i).index
        corrects[i] = np.take(corrects_all, idxs).tolist()
        predicted[i] = np.take(predicted_all, idxs).tolist()
        #predicted_probs[i] = np.take(predicted_all_probs, idxs).tolist()
        patient_ids[i] = np.take(patients, idxs).tolist()
    return {
        'predictions_from_start': predicted,
        'predictions_from_end': predicted,
        'patient_ids_from_start': patient_ids,
        'patient_ids_from_end': patient_ids,
        'predicted_probs_from_start': predicted, #predicted_probs,
        'predicted_probs_from_end': predicted, #predicted_probs,
        'labels': corrects,
        'labels_from_end': corrects
    }


def get_folds(patient_ids_dict, dataset, split='validation'):
    for i in range(len(patient_ids_dict)):
        yield dataset.index[dataset['patientid'].isin(patient_ids_dict[i]['train'])], dataset.index[
            dataset['patientid'].isin(patient_ids_dict[i][split])]


def print_cv_daily_result(cv_results, day):
    print(f'{day:2}'
          f' & {np.average(cv_results["test_hits"]):.1f}$\\pm${np.std(cv_results["test_hits"]):.4f}'
          f' & {np.average(cv_results["test_totals"]):.0f}'
          f' & {np.average(cv_results["test_acc"]):.4f}$\\pm${np.std(cv_results["test_acc"]):.4f}'
          f' & {np.average(cv_results["test_auc"]):.4f}$\\pm${np.std(cv_results["test_auc"]):.4f}'
          f' & {np.average(cv_results["test_sen"]):.4f}$\\pm${np.std(cv_results["test_sen"]):.4f}'
          f' & {np.average(cv_results["test_spe"]):.4f}$\\pm${np.std(cv_results["test_spe"]):.4f}'
          f' & {np.average(cv_results["test_f1"]):.4f}$\\pm${np.std(cv_results["test_f1"]):.4f}'
          )


def get_avg_by_key(obj, k):
    return np.average(
        [sum([elem * obj[kk]['test_totals'][i] for i, elem in enumerate(obj[kk][k])]) / sum(obj[kk]['test_totals']) for
         kk in obj.keys()])


def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance)


def get_std_by_key(obj, k):
    return np.average(
        [weighted_avg_and_std([elem for elem in obj[kk][k]], weights=obj[kk]['test_totals']) for kk in obj.keys()])


def print_global_metrics(best_model_scores):
    print(
        f'{get_avg_by_key(best_model_scores, "test_acc"):.4f}$\\pm${get_std_by_key(best_model_scores, "test_acc"):.4f}'
        f' & {get_avg_by_key(best_model_scores, "test_auc"):.4f}$\\pm${get_std_by_key(best_model_scores, "test_auc"):.4f}'
        f' & {get_avg_by_key(best_model_scores, "test_sen"):.4f}$\\pm${get_std_by_key(best_model_scores, "test_sen"):.4f}'
        f' & {get_avg_by_key(best_model_scores, "test_spe"):.4f}$\\pm${get_std_by_key(best_model_scores, "test_spe"):.4f}'
        f' & {get_avg_by_key(best_model_scores, "test_f1"):.4f}$\\pm${get_std_by_key(best_model_scores, "test_f1"):.4f}'
    )


def print_cv_by_fold(best_model_scores):
    for field in ['test_acc', 'test_sen']:
        fold_output = ''
        fold_data = [sum([elem * best_model_scores[kk]["test_totals"][i] for i, elem in
                          enumerate(best_model_scores[kk][field])]) / sum(best_model_scores[kk]["test_totals"]) for kk
                     in
                     best_model_scores.keys()]
        for i in range(5):
            if fold_output != '':
                fold_output = fold_output + f' & {fold_data[i]:.4f}'
            else:
                fold_output = f'{fold_data[i]:.4f}'
        print(fold_output)


classifier = SVC
model_params = []
# RF
if classifier == RandomForestClassifier:
    n_estimators = [5, 10, 25, 50, 75, 100, 125, 150, 175, 200]
    criterions = ["gini", "entropy"]
    min_samples_splits = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    class_weights = [None, 'balanced', 'balanced_subsample']
    for n_estimator in n_estimators:
        for criterion in criterions:
            for min_samples_split in min_samples_splits:
                for class_weight in class_weights:
                    model_params.append({'n_estimators': n_estimator, 'criterion': criterion,
                                         'min_samples_split': min_samples_split, 'class_weight': class_weight,
                                         'random_state': SEED, 'n_jobs': 10})

# SVC
elif classifier == SVC:
    cs = [0.3, 0.6, 1, 1.2, 1.4, 2, 3, 4]
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    tolerances = [1e-2, 1e-3, 1e-4]
    gammas = ["scale", "auto"]  # rbf, poly, sigmoid
    degrees = [2, 3, 4]  # only poly
    class_weights = [None, 'balanced']
    for c in cs:
        for class_weight in class_weights:
            for kernel in kernels:
                for tolerance in tolerances:
                    if kernel == 'linear':
                        model_params.append({'C': c, 'class_weight': class_weight, 'kernel': kernel, 'tol': tolerance,
                                             'random_state': SEED})
                    else:
                        for gamma in gammas:
                            if kernel == 'poly':
                                for degree in degrees:
                                    model_params.append({'C': c, 'class_weight': class_weight, 'kernel': kernel,
                                                         'tol': tolerance, 'gamma': gamma, 'degree': degree,
                                                         'random_state': SEED})
                            else:
                                model_params.append({'C': c, 'class_weight': class_weight, 'kernel': kernel,
                                                     'tol': tolerance, 'gamma': gamma, 'random_state': SEED})

# Load data from CSV files
data_static = pd.read_csv(os.path.join(HOSPITAL, DATASET_BASE_NAME, "dataset_static.csv"))
data_dynamic = pd.read_csv(os.path.join(HOSPITAL, DATASET_BASE_NAME, "dataset_dynamic.csv"))
labels = pd.read_csv(os.path.join(HOSPITAL, DATASET_BASE_NAME, "dataset_labels.csv"))

# Merge Static and Dynamic tables with labels
full_dataset = data_static.merge(data_dynamic, on='patientid').merge(labels, on='patientid')

top30path = os.path.join(HOSPITAL, f'{classifier.__name__.lower()}_{SELECT_METRIC}_top30.json')
if os.path.exists(top30path):
    with open(top30path, 'r') as file:
        best_grid_models = json.load(file)
else:
    with open(os.path.join(HOSPITAL, 'train_patient_ids.json'), 'r') as patient_ids_dict:
        patient_ids_dict = json.load(patient_ids_dict)

    data_train = full_dataset.loc[full_dataset['patientid'].isin(patient_ids_dict['train'])].drop(
        columns=['patientid', 'fallecimiento'])
    y_train = full_dataset.loc[full_dataset['patientid'].isin(patient_ids_dict['train'])]['fallecimiento']
    data_test = full_dataset.loc[full_dataset['patientid'].isin(patient_ids_dict['validation'])].drop(
        columns=['patientid', 'fallecimiento'])
    y_true_test = full_dataset.loc[full_dataset['patientid'].isin(patient_ids_dict['validation'])]['fallecimiento']
    grid_results = []
    for model_param in tqdm(model_params):
        clf = classifier(**model_param)
        clf.fit(data_train, y_train)
        # Skip validation data
        y_pred = clf.predict(data_test)
        grid_results.append((metric(y_true=y_true_test, y_pred=y_pred, **metric_params), model_param))

    best_grid_models = sorted(grid_results, key=lambda x: x[0])[::-1][:min(len(grid_results), 30)]
    with open(top30path, 'w') as outjson:
        json.dump(best_grid_models, outjson)

with open(os.path.join(HOSPITAL, 'patient_ids_cv.json'), 'r') as patient_ids_dict:
    patient_ids_dict = json.load(patient_ids_dict)

# Load data from CSV files
data_static = pd.read_csv(os.path.join(HOSPITAL, DATASET_NAME, 'dataset_static.csv'))
data_dynamic = pd.read_csv(os.path.join(HOSPITAL, DATASET_NAME, 'dataset_dynamic.csv'))
labels = pd.read_csv(os.path.join(HOSPITAL, DATASET_NAME, 'dataset_labels.csv'))

# Merge Static and Dynamic tables with labels
full_dataset = data_static.merge(data_dynamic, on='patientid').merge(labels, on='patientid')

specificity = make_scorer(compute_specificity)
hits = make_scorer(get_hits)
totals = make_scorer(get_totals)
f1 = make_scorer(get_f1)
scoring = {'acc': 'accuracy', 'sen': 'recall', 'spe': specificity, 'auc': 'roc_auc', 'hits': hits, 'totals': totals,
           'f1': f1}

# Save predictions
outpath = os.path.join(HOSPITAL, DATASET_NAME, 'output')
os.makedirs(outpath, exist_ok=True)

cv_validation_results = list()
for idx, result in tqdm(enumerate(best_grid_models)):
    clf = classifier(**{**result[1] })#,'probability': True})
    fold_generator = get_folds(patient_ids_dict=patient_ids_dict, dataset=full_dataset,
                               split='validation')
    _result = defaultdict(list)
    params = result[1]
    for fold_index in range(5):
        train_idxs, test_idxs = next(fold_generator)
        x_train, y_train, x_test, y_test = full_dataset.drop(['patientid', 'fallecimiento'], axis=1).iloc[train_idxs,
                                           :], full_dataset['fallecimiento'].iloc[train_idxs], full_dataset.drop(
            ['patientid', 'fallecimiento'], axis=1).iloc[
                                                                                               test_idxs, :], \
                                           full_dataset['fallecimiento'].iloc[test_idxs]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_probs = None#clf.predict_proba(x_test)
        with open(os.path.join(outpath, f'validation_{METRIC}_{idx}_{fold_index}.json'), 'w') as f:
            json.dump(get_fold_results(y_test, y_pred, y_pred_probs, full_dataset.iloc[test_idxs]), f)
        for k, v in scoring.items():
            _result[f'test_{k}'].append(get_scorer(v)(clf, x_test, y_test))
    cv_validation_results.append((_result, params))

best_valid_idxs = np.argsort([np.average(cv_result[0][SELECT_METRIC]) for cv_result in cv_validation_results])[::-1]
cv_test_results = list()
for idx, result in tqdm(enumerate(np.take(cv_validation_results, best_valid_idxs, axis=0))):
    clf = classifier(**{**result[1] })#,'probability': True})
    fold_generator = get_folds(patient_ids_dict=patient_ids_dict, dataset=full_dataset,
                               split='test')
    _result = defaultdict(list)
    params = result[1]
    for fold_index in range(5):
        train_idxs, test_idxs = next(fold_generator)
        x_train, y_train, x_test, y_test = full_dataset.drop(['patientid', 'fallecimiento'], axis=1).iloc[train_idxs,
                                           :], full_dataset['fallecimiento'].iloc[train_idxs], full_dataset.drop(
            ['patientid', 'fallecimiento'], axis=1).iloc[
                                                                                               test_idxs, :], \
                                           full_dataset['fallecimiento'].iloc[test_idxs]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_probs = None#clf.predict_proba(x_test)
        with open(os.path.join(outpath, f'test_{METRIC}_{best_valid_idxs[idx]}_{idx}_{fold_index}.json'), 'w') as f:
            json.dump(get_fold_results(y_test, y_pred, y_pred_probs, full_dataset.iloc[test_idxs]), f)
        for k, v in scoring.items():
            _result[f'test_{k}'].append(get_scorer(v)(clf, x_test, y_test))
    cv_test_results.append((_result, params))
best_model_scores, best_model_params = cv_test_results[best_valid_idxs[0]]

print("Best model params:")
pp.pprint(best_model_params)

# Print results
print("RESULTS:\n")

# By fold
for field in ["test_acc", "test_sen"]:
    fold_output = ''
    for fold in range(5):
        if fold_output != '':
            fold_output = fold_output + f' & {best_model_scores[field][fold]:.4f}'
        else:
            fold_output = f'{best_model_scores[field][fold]:.4f}'
    print(fold_output)
# Summary
print("\nMethod\tAccuracy\tAUC\tSensitivity\tSpecificity\tf1")
print(f'{np.average(best_model_scores["test_acc"]):.4f}$\\pm${np.std(best_model_scores["test_acc"]):.4f}'
      f' & {np.average(best_model_scores["test_auc"]):.4f}$\\pm${np.std(best_model_scores["test_auc"]):.4f}'
      f' & {np.average(best_model_scores["test_sen"]):.4f}$\\pm${np.std(best_model_scores["test_sen"]):.4f}'
      f' & {np.average(best_model_scores["test_spe"]):.4f}$\\pm${np.std(best_model_scores["test_spe"]):.4f}'
      f' & {np.average(best_model_scores["test_f1"]):.4f}$\\pm${np.std(best_model_scores["test_f1"]):.4f}'
      )

exit(0)

best_model_scores = dict()
for day in range(1, 21):  # Del ingreso hacía adelante
    partial_dataset = pd.DataFrame()

    # Generate partial dataset
    for patientid in data_dynamic['patientid'].unique():

        # print(patientid)

        # Dynamic events
        patient_events = data_dynamic.loc[data_dynamic['patientid'] == patientid]
        if day <= len(patient_events):  # Del ingreso hacía adelante
            events_until_day = patient_events.drop(columns='patientid').head(day)  # Del ingreso hacía adelante
            last_event = events_until_day.tail(1)
            last_event.reset_index(drop=True, inplace=True)
            events_until_day = events_until_day[
                events_until_day.columns.drop(list(events_until_day.filter(regex='_missing')))]

            partial_dynamic_data = pd.concat([events_until_day.min().add_prefix('min_'),
                                              events_until_day.max().add_prefix('max_'),
                                              events_until_day.mean().add_prefix('avg_')]
                                             ).to_frame().transpose()

            partial_dynamic_data.reset_index(drop=True, inplace=True)

            # Static part
            partial_static_data = data_static.loc[data_static['patientid'] == patientid]
            partial_static_data.reset_index(drop=True, inplace=True)

            # Join static and dynamic
            partial_dataset = partial_dataset.append(
                pd.concat([partial_static_data, last_event, partial_dynamic_data], axis=1))

    partial_dataset = partial_dataset.merge(labels, on='patientid')

    clf = classifier(**best_model_params)
    cv_results = cross_validate(clf,
                                partial_dataset.drop(columns=['patientid', 'fallecimiento']),
                                partial_dataset['fallecimiento'],
                                cv=get_folds(patient_ids_dict=patient_ids_dict, dataset=partial_dataset,
                                             split='test'),
                                n_jobs=10,
                                scoring=scoring)

    for i in range(5):
        for k in cv_results.keys():
            metric = best_model_scores.get(i, dict())
            metric_list = metric.get(k, list())
            metric_list.append(cv_results[k][i])
            metric[k] = metric_list
            best_model_scores[i] = metric

    print_cv_daily_result(cv_results, day)

print_global_metrics(best_model_scores)
print_cv_by_fold(best_model_scores)

best_model_scores = dict()
# From outcome
for day in range(20):
    partial_dataset = pd.DataFrame()

    # Generate partial dataset
    for patientid in data_dynamic['patientid'].unique():

        # print(patientid)

        # Dynamic events
        patient_events = data_dynamic.loc[data_dynamic['patientid'] == patientid]
        if day < len(patient_events):  # Del outcome hacía atrás
            if day == 0:  # Del outcome hacía atrás
                events_until_day = patient_events.drop(columns='patientid')
            else:
                events_until_day = patient_events.drop(columns='patientid').iloc[:-day]  # Del outcome hacía atrás

            last_event = events_until_day.tail(1)
            last_event.reset_index(drop=True, inplace=True)
            events_until_day = events_until_day[
                events_until_day.columns.drop(list(events_until_day.filter(regex='_missing')))]

            partial_dynamic_data = pd.concat([events_until_day.min().add_prefix('min_'),
                                              events_until_day.max().add_prefix('max_'),
                                              events_until_day.mean().add_prefix('avg_')]
                                             ).to_frame().transpose()

            partial_dynamic_data.reset_index(drop=True, inplace=True)

            # Static part
            partial_static_data = data_static.loc[data_static['patientid'] == patientid]
            partial_static_data.reset_index(drop=True, inplace=True)

            # Join static and dynamic
            partial_dataset = partial_dataset.append(
                pd.concat([partial_static_data, last_event, partial_dynamic_data], axis=1))

    partial_dataset = partial_dataset.merge(labels, on='patientid')
    clf = classifier(**best_model_params)
    cv_results = cross_validate(clf,
                                partial_dataset.drop(columns=['patientid', 'fallecimiento']),
                                partial_dataset['fallecimiento'],
                                cv=get_folds(patient_ids_dict=patient_ids_dict, dataset=partial_dataset,
                                             split='test'),
                                n_jobs=10,
                                scoring=scoring)

    for i in range(5):
        for k in cv_results.keys():
            metric = best_model_scores.get(i, dict())
            metric_list = metric.get(k, list())
            metric_list.append(cv_results[k][i])
            metric[k] = metric_list
            best_model_scores[i] = metric

    print_cv_daily_result(cv_results, day + 1)

print_global_metrics(best_model_scores)
print_cv_by_fold(best_model_scores)
