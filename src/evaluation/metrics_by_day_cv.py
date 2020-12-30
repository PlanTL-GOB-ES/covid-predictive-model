import os
import pickle
import copy
import json
import pprint
import numpy as np
from sklearn import metrics
from statistics import mean, stdev

all_accuracies = dict()
MAX_SEQ = 2
SCOPE = 20
FOLDS = 5
EXPERIMENTS_DIR = ''
BEST_DIR = ''
with open(BEST_DIR, 'r') as file:
    best_dir = json.load(file)

'''
Prints metrics by days with/without day filtering
'''

for root, dirs, _ in os.walk(EXPERIMENTS_DIR):
    for d in [d for d in dirs if d in best_dir]:
        print(f'MODEL "{d}"')
        fls = [file for file in os.listdir(os.path.join(root, d)) if "eval_preds_test" in file]
        fls_dates = np.argsort([os.path.getmtime(os.path.join(root, d, file)) for file in fls])[::-1][
                    :min(len(fls), FOLDS)]
        fls = [fls[idx] for idx in fls_dates]
        all_accuracies[d] = dict()
        for file in fls:
            raw_file = file
            all_accuracies[d][raw_file] = dict()
            file = os.path.join(root, d, file)

            data = json.load(open(file, "r"))

            predictions = dict()
            prediction_probs = dict()
            labels = dict()

            for step in data['predictions_from_start'].keys():
                lista_ids = data['patient_ids_from_start'][step]
                lista_labels = data['labels'][step]
                lista_predicciones = data['predictions_from_start'][step]
                lista_probs = data['predicted_probs_from_start'][step]

                for id, label, prediction, prob in zip(lista_ids, lista_labels, lista_predicciones, lista_probs):

                    if step == '0':
                        predictions[id] = []
                        prediction_probs[id] = []
                        labels[id] = label

                    predictions[id].append(prediction)
                    prediction_probs[id].append(prob)

            for min_sequence in range(1, MAX_SEQ):
                accuracies = []
                predictions_copy = copy.deepcopy(predictions)
                predictions_probs_copy = copy.deepcopy(prediction_probs)
                for days_before in range(0, SCOPE):
                    y_pred = []
                    y_pred_probs = []
                    y_true = []
                    for id in predictions_copy:
                        if len(predictions_copy[id]) >= min_sequence:
                            last_prediction = predictions_copy[id].pop(0)
                            y_pred.append(last_prediction)
                            y_pred_probs.append(predictions_probs_copy[id].pop(0))
                            y_true.append(labels[id])
                    if len(y_true):
                        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
                        specificity = tn / (tn + fp)
                        accuracies.append({'predicted': sum([1 for x in zip(y_true, y_pred) if x[0] == x[1]]),
                                           'total': len(y_true), 'accuracy': metrics.accuracy_score(y_true, y_pred),
                                           'auc': metrics.roc_auc_score(y_true, y_pred_probs),
                                           'sensitivity': metrics.recall_score(y_true, y_pred),
                                           'specificity': specificity,
                                           'f1': metrics.f1_score(y_true, y_pred, average='macro')})
                    # print(f'\t{days_before} days before: {acc:.3f} ({accuracy_score(y_true, y_pred, normalize=False)}/{len(y_true)})')

                day_acc = all_accuracies[d][raw_file]
                namesp = str(min_sequence)
                seq_acc = day_acc.get(namesp, list())
                seq_acc.append(copy.deepcopy(accuracies))
                day_acc[namesp] = seq_acc
                all_accuracies[d][raw_file] = day_acc

            for min_sequence in range(1, MAX_SEQ):
                accuracies = []
                predictions_copy = copy.deepcopy(predictions)
                predictions_probs_copy = copy.deepcopy(prediction_probs)
                for days_before in range(0, SCOPE):
                    y_pred = []
                    y_pred_probs = []
                    y_true = []
                    for id in predictions_copy:
                        if len(predictions_copy[id]) >= min_sequence:
                            last_prediction = predictions_copy[id].pop()
                            y_pred.append(last_prediction)
                            y_pred_probs.append(predictions_probs_copy[id].pop())
                            y_true.append(labels[id])
                    if len(y_true):
                        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
                        specificity = tn / (tn + fp)
                        accuracies.append({'predicted': sum([1 for x in zip(y_true, y_pred) if x[0] == x[1]]),
                                           'total': len(y_true), 'accuracy': metrics.accuracy_score(y_true, y_pred),
                                           'auc': metrics.roc_auc_score(y_true, y_pred_probs),
                                           'sensitivity': metrics.recall_score(y_true, y_pred),
                                           'specificity': specificity,
                                           'f1': metrics.f1_score(y_true, y_pred, average='macro')})
                    # print(f'\t{days_before} days before: {acc:.3f} ({accuracy_score(y_true, y_pred, normalize=False)}/{len(y_true)})')

                day_acc = all_accuracies[d][raw_file]
                namesp = str(min_sequence) + "_before"
                seq_acc = day_acc.get(namesp, list())
                seq_acc.append(copy.deepcopy(accuracies))
                day_acc[namesp] = seq_acc
                all_accuracies[d][raw_file] = day_acc

                # print(f'\n\tAverage: {mean(accuracies):.3f}')
                # print("---------\n")

metrics = dict()
for folder in all_accuracies.keys():
    file_key = list(all_accuracies[folder].keys())[0]
    seq_keys = all_accuracies[folder][file_key].keys()
    metrics[folder] = dict()
    for seq in seq_keys:
        metrics[folder][seq] = dict()
        for i in range(20):
            metric = {k: list() for k in all_accuracies[folder][file_key][seq][0][i].keys()}
            for file in all_accuracies[folder].keys():
                for k, v in all_accuracies[folder][file][seq][0][i].items():
                    metric[k].append(v)
            metric = {k: (mean(v), stdev(v)) for k, v in metric.items()}
            metrics[folder][seq][i] = metric

#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(metrics)
with open("./metrics.json", 'w') as outfile:
    json.dump(metrics, outfile)


def conditional_attachment(k, v):
    if k == 'predicted':
        return f'{v[0]:.1f}$\\pm${v[1]:.4f}'
    elif k == 'total':
        return f'{v[0]:.0f}'
    else:
        return f'{v[0]:.4f}$\\pm${v[1]:.4f}'


for metric_model in list(metrics.keys()):
    print(metric_model)
    for k in metrics[metric_model].keys():
        print(k)
        out = ''
        out2 = ''
        print(f'day & {" & ".join(list(metrics[metric_model][k].values())[0].keys())}')
        for kk, vv in metrics[metric_model][k].items():
            out = out + f'{kk + 1} & ' + ' & '.join([conditional_attachment(kkk, vvv) for kkk, vvv in vv.items()]) + '\n'
            out2 = out2 + f'{kk + 1}\t' + '\t'.join(
                [f'{vvv[0]}' if kkk == 'total' else f'{vvv[0]:.4f}\t{vvv[1]:.4f}' for kkk, vvv in vv.items()]) + '\n'
        print(out)
        print(out2)
