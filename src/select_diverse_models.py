import sys
import os
from glob import glob
import json
from statistics import mean
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import logging

MIN_ACC = 90
METRIC = 'accuracy'
assert METRIC in ['accuracy', 'f1']
VISUALIZE = False


def select_diverse_models(experiments_path, n_models, visualize=False, min_acc=90, metric='accuracy', log=logging.info,
                          use_cache=False, cache={}):
    figure = None
    if len(cache) == 0:
        accuracies = {}
        flattened_predictions = {}
        for experiment_path in os.listdir(experiments_path):
            try:
                if not os.path.isdir(os.path.join(experiments_path, experiment_path)):
                    continue
                if not os.path.exists(os.path.join(experiments_path, experiment_path, 'scores.csv')):
                    log(f'Skipping not finished experiment: {experiment_path}')
                    continue

                for valid_json_file in sorted(
                        glob(os.path.join(experiments_path, experiment_path) + '/eval_preds_valid_*json')
                ):
                    if experiment_path not in flattened_predictions:
                        flattened_predictions[experiment_path] = []
                    data = json.load(open(valid_json_file, 'r'))
                    for step in sorted(list(map(int, data['predictions_from_start'].keys()))):
                        step = str(step)
                        flattened_predictions[experiment_path].extend(data['predicted_probs_from_start'][step])
                for valid_csv_file in sorted(
                        glob(os.path.join(experiments_path, experiment_path) + '/eval_report_valid_*csv')
                ):
                    if experiment_path not in accuracies:
                        accuracies[experiment_path] = []
                    df = pd.read_csv(valid_csv_file, sep='\t')
                    accuracies[experiment_path].append(df[f'{metric} average weighted'][0])
                accuracies[experiment_path] = mean(list(map(lambda x: float(x[:-1]), accuracies[experiment_path])))
            except BaseException as e:
                if experiment_path in accuracies:
                    del accuracies[experiment_path]
                if experiment_path in flattened_predictions:
                    del flattened_predictions[experiment_path]
                log(f'Error: {experiment_path} {e}')

        log(f'Captured {len(accuracies)} paths')
        accuracies = {k: v for k, v in accuracies.items() if v > min_acc}
        flattened_predictions = {k: v for k, v in flattened_predictions.items() if k in accuracies}

        sorted_keys = sorted(accuracies, key=accuracies.get)
        sorted_keys.reverse()
        top_accuracy_model = sorted_keys[0]

        key_to_idx = {k: i for i, k in enumerate(accuracies.keys())}

        # Precompute similarity matrix
        for k in flattened_predictions:
            log(f'{k} {len(flattened_predictions[k])}')
        values = pd.DataFrame.from_dict(flattened_predictions).values.T
        prediction_similarity = distance_matrix(values, values, p=2)  # Euclidean

        if visualize:
            plt.matshow(prediction_similarity)
            figure = plt.figure()
            if use_cache:
                cache['figure'] = figure
    else:
        accuracies = cache['accuracies']
        prediction_similarity = cache['prediction_similarity']
        key_to_idx = cache['key_to_idx']
        if visualize:
            figure = cache['figure']
        selected_models = cache['selected_models']

    # Recursively find most different predictions
    selected_models = [top_accuracy_model] if len(cache) == 0 else selected_models
    while len(selected_models) < n_models:
        max_distance = 0
        max_key = ''
        for idx, model_to_select in enumerate(accuracies.keys()):
            distance = 0
            if model_to_select in selected_models:
                continue
            for selected_model in selected_models:
                distance += prediction_similarity[idx, key_to_idx[selected_model]]
            if distance >= max_distance:
                max_distance = distance
                max_key = model_to_select
        selected_models.append(max_key)
    if use_cache:
        cache['accuracies'] = accuracies
        cache['prediction_similarity'] = prediction_similarity
        cache['key_to_idx'] = key_to_idx
        cache['selected_models'] = selected_models
    return selected_models, accuracies, None if not visualize else figure


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python select_diverse_models.py EXPERIMENT_PATH N_MODELS')
    experiments_path = sys.argv[1]
    n_models = int(sys.argv[2])
    selected_models, accuracies, _ = select_diverse_models(experiments_path, n_models, visualize=VISUALIZE,
                                                           min_acc=MIN_ACC, metric=METRIC, log=print)
    print([(model, accuracies[model]) for model in selected_models])
    print('Copy this:')
    print(' '.join(os.path.join(experiments_path, model) for model in selected_models))
