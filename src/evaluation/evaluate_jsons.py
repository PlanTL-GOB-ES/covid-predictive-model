import argparse
from utils import NumpyEncoder
import os
import json
import logging
import time
from statistics import mean, stdev
from typing import List
from collections import defaultdict
from sklearn import metrics
from operator import add
import copy
from select_diverse_models import select_diverse_models
import decimal


def evaluate_jsons(experiments_paths: List[str], logistic_threshold: float,
             exp_dir: str, metric='accuracy', aggregate: str = 'add', aggregate_or_th: float = 0.5,
             n_folds: int = 5):
    assert aggregate in ['add', 'or']
    assert aggregate_or_th > 0 and aggregate_or_th <= 1
    assert metric in ['accuracy', 'f1']

    logging.info('Evaluating ensemble' if len(experiments_paths) > 1 else 'Evaluating single model')

    predictions = {'valid': [[defaultdict(list) for _ in range(len(experiments_paths))] for _ in range(n_folds)],
                   'test': [[defaultdict(list) for _ in range(len(experiments_paths))] for _ in range(n_folds)]}
    predicted_probs = {'valid': [[defaultdict(list) for _ in range(len(experiments_paths))] for _ in range(n_folds)],
                       'test': [[defaultdict(list) for _ in range(len(experiments_paths))] for _ in range(n_folds)]}
    labels = {'valid': [[] for _ in range(n_folds)], 'test': [[] for _ in range(n_folds)]}
    patient_ids_from_start = {'valid': [[] for _ in range(n_folds)], 'test': [[] for _ in range(n_folds)]}

    logging.info('Reading predictions')
    for idx, experiment_path in enumerate(experiments_paths):
        for subset in ['valid', 'test']:
            for fold in range(n_folds):
                json_name = f'eval_preds_{subset}_{fold}.json' if n_folds > 1 else f'eval_preds_{subset}.json'
                with open(os.path.join(experiment_path, json_name), 'r') as f:
                    data = json.load(f)
                predictions[subset][fold][idx] = data['predictions_from_start']
                predicted_probs[subset][fold][idx] = data['predicted_probs_from_start']
                if idx == 0:
                    labels[subset][fold] = data['labels']
                    patient_ids_from_start[subset][fold] = data['patient_ids_from_start']

    scores = {'valid': [], 'test': []}
    for fold in range(n_folds):
        for subset in ['valid', 'test']:
            subset_name = f'{subset}-{fold}' if n_folds > 1 else subset
            _, eval_acc = evaluate(predictions[subset][fold], labels[subset][fold],
                                   patient_ids_from_start[subset][fold], subset_name,
                                   logistic_threshold, exp_dir, metric=metric, aggregate=aggregate,
                                   aggregate_or_th=aggregate_or_th)
            scores[subset].append((_, eval_acc))
    logging.info(f'-{n_folds} Cross-validation\nVALID')

    def aggregate(metric):
        return dict(
            max=max(metric),
            min=min(metric),
            mean=mean(metric),
            std=stdev(metric) if len(metric) > 1 else 0.0
        )

    res = {}
    for _, metrics in scores['valid']:
        for e in metrics:
            if e not in res:
                res[e] = [metrics[e]]
            else:
                res[e].append(metrics[e])
    for e in res:
        res[e] = aggregate(res[e])
    logging.info('Metrics: ' + str(res))
    valid_metrics = res

    logging.info('TEST')
    res = {}
    for _, metrics in scores['test']:
        for e in metrics:
            if e not in res:
                res[e] = [metrics[e]]
            else:
                res[e].append(metrics[e])
    for e in res:
        res[e] = aggregate(res[e])
    test_metrics = res
    logging.info('Metrics: ' + str(res))
    return valid_metrics, test_metrics


def evaluate(predictions, corrects, patient_ids_from_start, subset_name: str, logistic_threshold: float,
             exp_dir: str, metric='accuracy', aggregate: str = 'add', aggregate_or_th: float = 0.5):
    metric_name = metric
    if metric == 'accuracy':
        metric_score = metrics.accuracy_score
        metric_args = {}
        metric_other_name, metric_other_args, metric_other_score = 'f1', {'average': 'macro'}, metrics.f1_score
    else:
        metric_score = metrics.f1_score
        metric_args = {'average': 'macro'}
        metric_other_name, metric_other_args, metric_other_score = 'accuracy', {}, metrics.accuracy_score

    # compute predictions and from end (right-aligned sequences) using the sequence length for each prediction
    max_steps = len(predictions[0].keys())

    # Compute voted predictions
    def aggregate_or(votes):
        return (1 if len(list(filter(lambda x: x == 1, votes))) / len(votes) >= aggregate_or_th else 0,
                sum(votes) / len(votes))

    predicted = defaultdict(list)
    predicted_probs = defaultdict(list)
    for step in range(max_steps):
        step = str(step)
        # for each step, sum the prediction of each model in the ensemble
        preds_votes = []
        if aggregate == 'add':
            for model_idx in range(len(predictions)):
                if len(preds_votes) == 0:
                    preds_votes = [pred for pred in predictions[model_idx][step]]
                else:
                    preds_votes = list(map(
                        add, preds_votes, [pred for pred in predictions[model_idx][step]]))

            predicted[step] = [1 if pred >= logistic_threshold * len(predictions) else 0 for pred in preds_votes]
            predicted_probs[step] = preds_votes
        else:
            preds_votes_to_aggregate = []
            for model_idx in range(len(predictions)):
                if len(preds_votes_to_aggregate) == 0:
                    preds_votes_to_aggregate = [pred for pred in predictions[model_idx][step]]
                    preds_votes_to_aggregate = [[1 if pred >= logistic_threshold else 0 for pred in
                                                 preds_votes_to_aggregate]]
                else:
                    new_votes = [pred for pred in predictions[model_idx][step]]
                    new_votes = [1 if pred >= logistic_threshold else 0 for pred in new_votes]
                    preds_votes_to_aggregate.append(new_votes)
            pred_probs_or = []
            for idx_pred_ in range(len(preds_votes_to_aggregate[0])):
                preds_votes.append(aggregate_or([preds[idx_pred_] for preds in preds_votes_to_aggregate]))
            for idx_pred_vote, pred_vote in enumerate(preds_votes):
                decision, probs = pred_vote
                preds_votes[idx_pred_vote] = decision
                pred_probs_or.append(probs)
            predicted[step] = preds_votes
            predicted_probs[step] = pred_probs_or

    predictions = dict()
    prediction_probs = dict()
    labels = dict()
    for step in predicted.keys():
        step = str(step)
        lista_ids = patient_ids_from_start[step]
        lista_labels = corrects[step]
        lista_predicciones = predicted[step]
        lista_probs = predicted_probs[step]

        for id, label, prediction, prob in zip(lista_ids, lista_labels, lista_predicciones, lista_probs):
            if step == '0':
                predictions[id] = []
                prediction_probs[id] = []
                labels[id] = label

            predictions[id].append(prediction)
            prediction_probs[id].append(prob)

    predicted_from_end = defaultdict(list)
    predicted_probs_from_end = defaultdict(list)
    patient_ids_from_end = defaultdict(list)
    corrects_from_end = defaultdict(list)
    predictions_copy = copy.deepcopy(predictions)
    predictions_probs_copy = copy.deepcopy(prediction_probs)
    for step in range(max_steps):
        step = str(step)
        y_pred = []
        y_pred_probs = []
        y_true = []
        patient_ids_step = []
        for id in predictions_copy:
            if len(predictions_copy[id]) > 0:
                last_prediction = predictions_copy[id].pop()
                y_pred.append(last_prediction)
                y_pred_probs.append(predictions_probs_copy[id].pop())
                y_true.append(labels[id])
                patient_ids_step.append(id)
        patient_ids_from_end[step] = patient_ids_step
        predicted_from_end[step] = y_pred
        predicted_probs_from_end[step] = y_pred_probs
        corrects_from_end[step] = y_true

    # write to disk predictions and corrects labels
    eval_preds = {"predictions_from_start": predicted,
                  "predictions_from_end": predicted_from_end,
                  "patient_ids_from_start": patient_ids_from_start,
                  "patient_ids_from_end": patient_ids_from_end,
                  "predicted_probs_from_start": predicted_probs,
                  "predicted_probs_from_end": predicted_probs_from_end,
                  "labels": corrects,
                  "labels_from_end": corrects_from_end}

    with open(os.path.join(exp_dir, 'eval_preds_' + subset_name + '.json'), 'w') as pn:
        json.dump(eval_preds, pn, cls=NumpyEncoder)

    # Compute evaluations metrics and write report
    eval_metrics = {"from_start": defaultdict(), "from_end": defaultdict(),
                    f"{metric_name}_avg_weighted": defaultdict()}
    for step in range(max_steps):
        step = str(step)
        # mean over all the correct predictions at given step
        assert (len(predicted[step]) == len(corrects[step]) and len(predicted_from_end[step]) == len(
            corrects_from_end[step])), \
            'number of labels different from number of predictions'

        eval_metrics["from_start"][step] = {metric_name: metric_score(corrects[step], predicted[step], **metric_args),
                                            metric_other_name: metric_other_score(corrects[step], predicted[step],
                                                                                  **metric_other_args),
                                            "sensitivity": metrics.recall_score(corrects[step], predicted[step]),
                                            "corrects":
                                                f'{metrics.accuracy_score(corrects[step], predicted[step], normalize=False)}',
                                            "examples":
                                                f'{len(predicted[step])}'}

        eval_metrics["from_end"][step] = {
            metric_name: metric_score(corrects_from_end[step], predicted_from_end[step], **metric_args),
            metric_other_name: metric_other_score(corrects_from_end[step], predicted_from_end[step],
                                                  **metric_other_args),
            "sensitivity": metrics.recall_score(corrects_from_end[step], predicted_from_end[step]),
            "corrects":
                f'{metrics.accuracy_score(corrects_from_end[step], predicted_from_end[step], normalize=False)}',
            "examples": f'{len(predicted_from_end[step])}'}

    predicted_all_scores = []
    for step in range(max_steps):
        step = str(step)
        predicted_all_scores.extend(predicted_probs[step])

    predicted_all = []
    for step in range(max_steps):
        step = str(step)
        predicted_all.extend(predicted[step])

    predicted_all_from_end = []
    for step in range(max_steps):
        step = str(step)
        predicted_all_from_end.extend(predicted_from_end[step])

    corrects_all = []
    for step in range(max_steps):
        step = str(step)
        corrects_all.extend(corrects[step])

    corrects_all_from_end = []
    for step in range(max_steps):
        step = str(step)
        corrects_all_from_end.extend(corrects_from_end[step])

    eval_metrics[f"{metric_name}_avg_weighted"] = metric_score(corrects_all, predicted_all, **metric_args)
    eval_metrics[f"{metric_name}_avg_weighted_from_end"] = metric_score(corrects_all_from_end, predicted_all_from_end,
                                                                        **metric_args)
    eval_metrics[f"{metric_other_name}_avg_weighted"] = metric_other_score(corrects_all, predicted_all,
                                                                           **metric_other_args)
    eval_metrics[f"{metric_other_name}_avg_weighted_from_end"] = metric_other_score(corrects_all_from_end,
                                                                                    predicted_all_from_end,
                                                                                    **metric_other_args)
    eval_metrics["auc"] = metrics.roc_auc_score(corrects_all, predicted_all_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(corrects_all, predicted_all).ravel()
    specificity = tn / (tn + fp)
    eval_metrics["sensitivity"] = metrics.recall_score(corrects_all, predicted_all)
    eval_metrics["sensitivity_from_end"] = metrics.recall_score(corrects_all_from_end, predicted_all_from_end)
    eval_metrics["specificity"] = specificity

    # TODO: encapsulate the evaluation report in a function
    eval_report = '\t'.join(['days from hospitalization',
                             'corrects',
                             'examples',
                             f'{metric_name} per day',
                             f'{metric_name} average weighted',
                             f'{metric_other_name} per day',
                             f'{metric_other_name} average weighted',
                             'sensitivity per day',
                             'sensitivity average'
                             ])

    for step in range(max_steps):
        step = str(step)
        eval_report += '\t'.join([f'\n{step}',
                                  f'{eval_metrics["from_start"][step]["corrects"]}',
                                  f'{eval_metrics["from_start"][step]["examples"]}',
                                  f'{eval_metrics["from_start"][step][f"{metric_name}"] * 100:.2f}%',
                                  f'{eval_metrics[f"{metric_name}_avg_weighted"] * 100:.2f}%',
                                  f'{eval_metrics["from_start"][step][f"{metric_other_name}"] * 100:.2f}%',
                                  f'{eval_metrics[f"{metric_other_name}_avg_weighted"] * 100:.2f}%',
                                  f'{eval_metrics["from_start"][step]["sensitivity"] * 100:.2f}%',
                                  f'{eval_metrics["sensitivity"] * 100:.2f}%'
                                  ])

    eval_report += '\n'
    eval_report += '\t'.join(['days before discharge',
                              'corrects',
                              'examples',
                              f'{metric_name} per day',
                              f'{metric_name} average weighted',
                              f'{metric_other_name} per day',
                              f'{metric_other_name} average weighted',
                              'sensitivity per day',
                              'sensitivity average'])
    for step in range(max_steps):
        step = str(step)
        eval_report += '\t'.join([f'\n{step}',
                                  f'{eval_metrics["from_end"][step]["corrects"]}',
                                  f'{eval_metrics["from_end"][step]["examples"]}',
                                  f'{eval_metrics["from_end"][step][f"{metric_name}"] * 100:.2f}%',
                                  f'{eval_metrics[f"{metric_name}_avg_weighted_from_end"] * 100:.2f}%',
                                  f'{eval_metrics["from_end"][step][f"{metric_other_name}"] * 100:.2f}%',
                                  f'{eval_metrics[f"{metric_other_name}_avg_weighted_from_end"] * 100:.2f}%',
                                  f'{eval_metrics["from_end"][step]["sensitivity"] * 100:.2f}%',
                                  f'{eval_metrics["sensitivity_from_end"] * 100:.2f}%'])

    logging.info(eval_report)
    with open(os.path.join(exp_dir, 'eval_report_' + subset_name + '.csv'), 'w') as fn:
        fn.writelines(eval_report)
    logging.info(
        f"{metric_name.upper()} GLOBAL {subset_name}: " + f'{eval_metrics[f"{metric_name}_avg_weighted"] * 100:.4f}%')

    output_table = {
        f'{metric_name}_avg_weighted': eval_metrics[f'{metric_name}_avg_weighted'],
        f'{metric_other_name}_avg_weighted': eval_metrics[f'{metric_other_name}_avg_weighted'],
        'sensitivity': eval_metrics['sensitivity'],
        'specificity': eval_metrics['specificity'],
        'auc': eval_metrics['auc']
    }

    return None, output_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model from jsons')
    parser.add_argument('models_path', type=str, help='Path to model directory. If more than one path is provided, an'
                                                      'ensemble of models is loaded', nargs='+')
    parser.add_argument('--logistic-threshold', type=float, help='Threshold of the logistic regression (default 0.5)',
                        default=0.5)
    parser.add_argument('--cross-val', type=int, help='K-Fold cross validation', default=5)
    parser.add_argument('--nested-cross-val', type=int, help='Nested K-Fold cross validation',
                        default=-1)
    parser.add_argument('--max-seq', type=int, help='Maximum sequence length (longer sequences are truncated from the'
                                                    'end default: -1 -> do not truncate)', default=-1)
    parser.add_argument('--aggregate', type=str, help='Voting for ensembles: add -> sum raw probabilities,'
                                                      'or -> count discretized predictions, return 1 if # models'
                                                      'saying 1 >= aggregate-or-th', default='add')
    parser.add_argument('--aggregate-or-th', type=float,
                        help='Threshold to aggregate an ensemble (only if --aggregate or)',
                        default=0.5)
    parser.add_argument('--metric', type=str, choices=['accuracy', 'f1'], help='Metric',
                        default='accuracy')  # IGNORE this argument (TODO: remove)
    parser.add_argument('--optimize', type=str, choices=['no', 'logistic-threshold', 'aggregate-or-th'],
                        help='Whether to optimize a specific variable',
                        default='no')
    parser.add_argument('--select-ensemble', action='store_true', help='Select how many (and which of them) models to'
                                                                       'ensemble. In this case, models_path must'
                                                                       'be the path containing the experiments.')
    parser.add_argument('--search-step-size', type=float,
                        help='Step size (float) if --optimize != no',
                        default=0.1)
    parser.add_argument('--visualize-diverse-models', action='store_true', help='Visualize similarity matrix'
                                                                                'of ensemble')
    parser.add_argument('--search-max-models', type=int, help='If --select-ensemble, max models to select')
    parser.add_argument('--metric-models-ensemble', type=str, choices=['accuracy', 'f1'], help='Metric to select'
                                                                                               'models for ensemble,'
                                                                                               'if --select-ensemble.',
                        default='accuracy')
    parser.add_argument('--select-ensemble-th', type=int, default=90, help='Minimum --metric-models-ensemble,'
                                                                           '0-100.')
    args_eval = parser.parse_args()

    round_th = -1*decimal.Decimal(str(args_eval.search_step_size)).as_tuple().exponent

    if args_eval.nested_cross_val != -1:
        raise NotImplementedError('Nested cross-validation is not implemented')

    if args_eval.max_seq != -1:
        raise NotImplementedError('Max-seq is not implemented')

    if args_eval.search_max_models and not args_eval.select_ensemble:
        raise RuntimeError('If --search-max-models is defined, --select-ensemble must be set.')

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    path = os.path.join('..', 'evaluations', f'eval-{timestamp}')
    os.makedirs(path, exist_ok=True)
    log_path = os.path.join(path, 'eval.log')

    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    logging.info(args_eval)

    if args_eval.optimize or args_eval.select_ensemble:
        all_results = {}

    def count_models(path_):
        return len([experiment for experiment in os.listdir(path_) if os.path.isdir(os.path.join(path_, experiment))])

    if not args_eval.select_ensemble and args_eval.optimize == 'no':
        evaluate_jsons(experiments_paths=args_eval.models_path, logistic_threshold=args_eval.logistic_threshold,
                       exp_dir=path, metric=args_eval.metric, aggregate=args_eval.aggregate,
                       aggregate_or_th=args_eval.aggregate_or_th, n_folds=args_eval.cross_val)

    elif not args_eval.select_ensemble and args_eval.optimize != 'no':
        f1 = None
        new_f1 = None
        best_f1 = None
        logistic_threshold = args_eval.logistic_threshold
        aggregate_or_th = args_eval.aggregate_or_th
        if args_eval.optimize == 'aggregate-or-th' and args_eval.aggregate != 'or':
            raise RuntimeError('If --optimize ensemble, --aggregate must be set to or')
        # TODO: Binary search with tolerance?
        history_metrics = []
        history_arguments = []
        i_search = 0
        all_results[args_eval.optimize] = {}
        # while logistic_threshold > 0 if args_eval.optimize == 'logistic-threshold' else aggregate_or_th > 0:
        # while f1 is None or new_f1 is None or new_f1 >= f1 and (
        #        round(logistic_threshold if args_eval.optimize == 'logistic-threshold' else aggregate_or_th,
        #             round_th) > 0):
        # COMMENTED out due to the deprecation of early stopping etc
        while (logistic_threshold if args_eval.optimize == 'logistic-threshold' else aggregate_or_th) > 0:
            logging.info(f'''{args_eval.optimize} {logistic_threshold if args_eval.optimize == 'logistic-threshold' else
                             aggregate_or_th}''')
            valid_metrics, test_metrics = evaluate_jsons(experiments_paths=args_eval.models_path,
                                                         logistic_threshold=logistic_threshold,
                                                         exp_dir=path, metric=args_eval.metric,
                                                         aggregate=args_eval.aggregate,
                                                         aggregate_or_th=aggregate_or_th, n_folds=args_eval.cross_val)
            history_metrics.append(history_metrics)
            history_arguments.append(
                logistic_threshold if args_eval.optimize == 'logistic-threshold' else aggregate_or_th
            )
            if new_f1 is not None:
                f1 = new_f1
            new_f1 = valid_metrics['f1_avg_weighted']['mean']
            if best_f1 is None or new_f1 > best_f1:
                best_f1 = new_f1
                best_i_search = i_search

            if args_eval.optimize == 'logistic-threshold':
                all_results[args_eval.optimize][logistic_threshold] = {'valid': new_f1,
                                                                       'test': test_metrics['f1_avg_weighted']['mean']}
                logistic_threshold -= args_eval.search_step_size
                logistic_threshold = round(logistic_threshold, round_th)
            else:
                all_results[args_eval.optimize][aggregate_or_th] = {'valid': new_f1,
                                                                    'test': test_metrics['f1_avg_weighted']['mean']}
                aggregate_or_th -= args_eval.search_step_size
                aggregate_or_th = round(aggregate_or_th, round_th)
            i_search += 1
        logging.info('_________')
        logging.info('BEST FOUND')
        logging.info(f'Best configuration: {args_eval.optimize}  = {history_arguments[best_i_search]}')
        logging.info('With results:')
        if args_eval.optimize == 'logistic-threshold':
            logistic_threshold = history_arguments[best_i_search]
        else:
            aggregate_or_th = history_arguments[best_i_search]
        evaluate_jsons(experiments_paths=args_eval.models_path,
                       logistic_threshold=logistic_threshold,
                       exp_dir=path, metric=args_eval.metric, aggregate=args_eval.aggregate,
                       aggregate_or_th=aggregate_or_th, n_folds=args_eval.cross_val)

    elif args_eval.select_ensemble and len(args_eval.models_path) != 1:
        raise RuntimeError('If --select-ensemble is set, models_path must be the path containing the experiments.')
    elif args_eval.select_ensemble and args_eval.optimize == 'no':
        previous_f1 = 0
        best_selected_models = []
        best_plot = None
        stop_empty = False
        all_results['n_models'] = {}
        for n_models in range(1, count_models(args_eval.models_path[0])):
            n_models += 1
            if args_eval.search_max_models and n_models > args_eval.search_max_models:
                break
            logging.info(f'{n_models} models:')
            selected_models, _, plot_sim = select_diverse_models(args_eval.models_path[0], n_models,
                                                                 visualize=args_eval.visualize_diverse_models,
                                                                 use_cache=True,
                                                                 metric=args_eval.metric_models_ensemble,
                                                                 min_acc=args_eval.select_ensemble_th)
            selected_models = selected_models.copy()
            for idx, e in enumerate(selected_models):
                if len(e) == 0:
                    stop_empty = True
                    break
                selected_models[idx] = os.path.join(args_eval.models_path[0], e)
            if stop_empty:
                logging.info('WARNING: Early stopping search due to EMPTY MODEL: ' + e)
                break
            logging.info(selected_models)
            valid_metrics, test_metrics = evaluate_jsons(experiments_paths=selected_models,
                                                         logistic_threshold=args_eval.logistic_threshold,
                                                         exp_dir=path, metric=args_eval.metric,
                                                         aggregate=args_eval.aggregate,
                                                         aggregate_or_th=args_eval.aggregate_or_th,
                                                         n_folds=args_eval.cross_val)
            all_results['n_models'][n_models] = {'valid': valid_metrics['f1_avg_weighted']['mean'],
                                                 'test': test_metrics['f1_avg_weighted']['mean']}
            '''
            # COMMENTED out due to the deprecation of early stopping etc
            if valid_metrics['f1_avg_weighted']['mean'] <= previous_f1:  # TODO: Check < ?
                break
            else:
                best_plot = plot_sim
                best_selected_models = selected_models
            '''
            if valid_metrics['f1_avg_weighted']['mean'] > previous_f1:
                best_plot = plot_sim
                best_selected_models = selected_models
        logging.info('_________')
        logging.info('BEST FOUND')
        logging.info('Best configuration:')
        logging.info(f'{len(best_selected_models)} models')
        logging.info('Selected models:')
        logging.info(' '.join(os.path.join(args_eval.models_path[0], model) for model in best_selected_models))
        logging.info('With results:')
        evaluate_jsons(experiments_paths=best_selected_models,
                       logistic_threshold=args_eval.logistic_threshold,
                       exp_dir=path, metric=args_eval.metric, aggregate=args_eval.aggregate,
                       aggregate_or_th=args_eval.aggregate_or_th, n_folds=args_eval.cross_val)
        if args_eval.visualize_diverse_models:
            best_plot.savefig(os.path.join(path, 'similarity_models.png'))
    else:
        assert args_eval.select_ensemble and args_eval.optimize == 'aggregate-or-th'

        previous_f1 = 0
        best_selected_models = []
        logging.info(count_models(args_eval.models_path[0]))
        stop_empty = False
        all_results['n_models'] = {}
        for n_models in range(1, count_models(args_eval.models_path[0])):
            n_models += 1
            if args_eval.search_max_models and n_models > args_eval.search_max_models:
                break
            logging.info(f'{n_models} models:')
            selected_models, _, plot_sim = select_diverse_models(args_eval.models_path[0], n_models,
                                                                 visualize=args_eval.visualize_diverse_models,
                                                                 use_cache=True,
                                                                 metric=args_eval.metric_models_ensemble,
                                                                 min_acc=args_eval.select_ensemble_th)
            selected_models = selected_models.copy()
            for idx, e in enumerate(selected_models):
                if len(e) == 0:
                    stop_empty = True
                    break
                selected_models[idx] = os.path.join(args_eval.models_path[0], e)
            if stop_empty:
                break
            if n_models not in all_results['n_models']:
                all_results['n_models'][n_models] = {}
            logging.info(selected_models)
            f1 = None
            new_f1 = None
            best_f1 = None
            logistic_threshold = args_eval.logistic_threshold
            aggregate_or_th = args_eval.aggregate_or_th
            if args_eval.optimize == 'ensemble' and args_eval.aggregate != 'or':
                raise RuntimeError('If --optimize ensemble, --aggregate must be set to or')
            history_metrics = []
            history_arguments = []
            i_search = 0
            # while logistic_threshold > 0 if args_eval.optimize == 'logistic-threshold' else aggregate_or_th > 0:
            # COMMENTED out due to the deprecation of early stopping etc
            #while f1 is None or new_f1 is None or new_f1 >= f1 and aggregate_or_th > 0:
            while aggregate_or_th > 0:
                # logging.info(
                #    f'''{args_eval.optimize} {round(logistic_threshold, 1) if args_eval.optimize == 'logistic-threshold' else round(
                #        aggregate_or_th, 1)}''')
                logging.info(f'{args_eval.optimize} = {aggregate_or_th}')
                valid_metrics, test_metrics = evaluate_jsons(experiments_paths=selected_models,
                                                             logistic_threshold=logistic_threshold,
                                                             exp_dir=path, metric=args_eval.metric,
                                                             aggregate=args_eval.aggregate,
                                                             aggregate_or_th=aggregate_or_th,
                                                             n_folds=args_eval.cross_val)
                history_metrics.append(history_metrics)
                history_arguments.append(
                    aggregate_or_th
                )
                if new_f1 is not None:
                    f1 = new_f1
                new_f1 = valid_metrics['f1_avg_weighted']['mean']
                if args_eval.optimize not in all_results['n_models'][n_models]:
                    all_results['n_models'][n_models][args_eval.optimize] = {}
                all_results['n_models'][n_models][args_eval.optimize][aggregate_or_th] = \
                    {'valid': valid_metrics['f1_avg_weighted']['mean'], 'test': test_metrics['f1_avg_weighted']['mean']}
                if best_f1 is None or new_f1 > best_f1:
                    best_f1 = new_f1
                    best_i_search = i_search
                aggregate_or_th -= args_eval.search_step_size
                aggregate_or_th = round(aggregate_or_th, round_th)
                i_search += 1
            '''
            # COMMENTED out due to the deprecation of early stopping etc
            if best_f1 <= previous_f1:  # TODO: Check: < ?
                break
            else:
                best_selected_models = selected_models
                best_aggregate_or_th = history_arguments[best_i_search]
                previous_f1 = best_f1
                best_plot = plot_sim
            '''
            if best_f1 > previous_f1:
                best_selected_models = selected_models
                best_aggregate_or_th = history_arguments[best_i_search]
                previous_f1 = best_f1
                best_plot = plot_sim
        logging.info('_________')
        logging.info('BEST FOUND')
        logging.info(f'Best configuration: {args_eval.optimize} = {best_aggregate_or_th}')
        logging.info(f'{len(best_selected_models)} models')
        logging.info(best_selected_models)
        logging.info('With results:')
        aggregate_or_th = best_aggregate_or_th
        evaluate_jsons(experiments_paths=best_selected_models,
                       logistic_threshold=args_eval.logistic_threshold,
                       exp_dir=path, metric=args_eval.metric, aggregate=args_eval.aggregate,
                       aggregate_or_th=best_aggregate_or_th, n_folds=args_eval.cross_val)
        if args_eval.visualize_diverse_models:
            best_plot.savefig(os.path.join(path, 'similarity_models.png'))

    if args_eval.optimize or args_eval.select_ensemble:
        with open(os.path.join(path, 'all_results_optimize_eval.json'), 'w') as f:
            json.dump(all_results, f, indent=4, sort_keys=True)
        logging.info(f"Saved optimization results in {os.path.join(path, 'all_results_optimize_eval.json')}")
