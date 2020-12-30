import os
import pickle
import copy
import json
import pprint
import numpy as np
from statistics import mean, stdev

all_accuracies = dict()
MAX_SEQ = 6
SCOPE = 20
FOLDS = 5
EXPERIMENTS_DIR = ''
BEST_DIR = ''
with open(BEST_DIR, 'r') as file:
    best_dir = json.load(file)

for root, dirs, _ in os.walk(EXPERIMENTS_DIR):
    for d in [d for d in dirs if d in best_dir]:
        print(f'MODEL "{d}"')
        fls = [file for file in os.listdir(os.path.join(root, d)) if "eval_preds_" in file]
        fls_dates = np.argsort([os.path.getmtime(os.path.join(root, d, file)) for file in fls])[::-1][:min(len(fls), FOLDS)]
        fls = [fls[idx] for idx in fls_dates]
        all_accuracies[d] = dict()
        for file in fls:
            file = os.path.join(root, d, file)

            data = pickle.load(open(file,"rb"))

            predictions = dict()
            labels = dict()

            for step in data['predictions_from_start'].keys():
                lista_ids = data['patient_ids_from_start'][step]
                lista_labels = data['labels'][step]
                lista_predicciones = data['predictions_from_start'][step]

                for id, label, prediction in zip(lista_ids, lista_labels, lista_predicciones):

                    if step == 0:
                        predictions[id] = []
                        labels[id] = label

                    predictions[id].append(prediction)

            for min_sequence in range(1, MAX_SEQ):
                accuracies = []
                predictions_copy = copy.deepcopy(predictions)
                for days_before in range(0, SCOPE):
                    y_pred = []
                    y_true = []
                    for id in predictions_copy:
                        if len(predictions_copy[id]) >= min_sequence:
                            last_prediction = predictions_copy[id].pop(0)
                            y_pred.append(last_prediction)
                            y_true.append(labels[id])
                    if len(y_true):
                        accuracies.append({'predicted': sum([1 for x in zip(y_true, y_pred) if x[0] == x[1]]), 'total': len(y_true)})
                    # print(f'\t{days_before} days before: {acc:.3f} ({accuracy_score(y_true, y_pred, normalize=False)}/{len(y_true)})')

                day_acc = all_accuracies[d]
                namesp = str(min_sequence)
                seq_acc = day_acc.get(namesp, list())
                seq_acc.append(copy.deepcopy(accuracies))
                day_acc[namesp] = seq_acc
                all_accuracies[d] = day_acc


            for min_sequence in range(1, MAX_SEQ):
                accuracies = []
                predictions_copy = copy.deepcopy(predictions)
                for days_before in range(0, SCOPE):
                    y_pred = []
                    y_true = []
                    for id in predictions_copy:
                        if len(predictions_copy[id]) >= min_sequence:
                            last_prediction = predictions_copy[id].pop()
                            y_pred.append(last_prediction)
                            y_true.append(labels[id])
                    if len(y_true):
                        accuracies.append({'predicted': sum([1 for x in zip(y_true, y_pred) if x[0] == x[1]]),
                                           "y_pred": y_pred, "y_true": y_true,
                                           'total': len(y_true)})
                    # print(f'\t{days_before} days before: {acc:.3f} ({accuracy_score(y_true, y_pred, normalize=False)}/{len(y_true)})')

                day_acc = all_accuracies[d]
                namesp = str(min_sequence) + "_before"
                seq_acc = day_acc.get(namesp, list())
                seq_acc.append(copy.deepcopy(accuracies))
                day_acc[namesp] = seq_acc
                all_accuracies[d] = day_acc

                # print(f'\n\tAverage: {mean(accuracies):.3f}')
                # print("---------\n")


pp = pprint.PrettyPrinter(indent=4)
pp.pprint(all_accuracies)
all_accuracies = {k:{k2:(mean([sum([f['predicted'] for f in fold])/sum([f['total'] for f in fold]) for fold in v2]),
                         stdev([sum([f['predicted'] for f in fold])/sum([f['total'] for f in fold]) for fold in v2]))
                     for k2, v2 in v.items()} for k, v in all_accuracies.items()}
pp.pprint(all_accuracies)

for k in all_accuracies.keys():
    l_dat = list()
    l_dat_bef = list()

    for kk in all_accuracies[k].keys():
        if '_before' in kk:
            l_dat_bef.append(all_accuracies[k][kk])
        else:
            l_dat.append(all_accuracies[k][kk])

    for i in range(0, len(l_dat)):
        print(f'{i} & {l_dat[i][0]:.4f}$\\pm${l_dat[i][1]:.4f} & {l_dat_bef[i][0]:.4f}$\\pm${l_dat_bef[i][1]:.4f}')

with open("./best_models.json", 'w') as outfile:
    json.dump(all_accuracies, outfile)
