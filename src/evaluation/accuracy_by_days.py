import os
import pickle
import copy
from sklearn.metrics import accuracy_score, roc_curve, auc
from statistics import mean
from heapq import nlargest
import json
import pprint
from matplotlib import pyplot as plt

all_accuracies = dict()
MAX_SEQ = 6
for root, dirs, _ in os.walk('../../covid-predictive-model/experiments5/'):
    for d in dirs:
        print(f'MODEL "{d}"')
        file = os.path.join(root, d, 'eval_preds.pickle')

        data = pickle.load(open(file, "rb"))

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

            for days_before in range(0, MAX_SEQ):
                predictions_copy = copy.deepcopy(predictions)
                y_pred = []
                y_true = []
                for id in predictions_copy:
                    if len(predictions_copy[id]) >= min_sequence:
                        last_prediction = predictions_copy[id].pop()
                        y_pred.append(last_prediction)
                        y_true.append(labels[id])
                acc = accuracy_score(y_true, y_pred)
                accuracies.append(acc)
                # print(f'\t{days_before} days before: {acc:.3f} ({accuracy_score(y_true, y_pred, normalize=False)}/{len(y_true)})')

            seq_acc = all_accuracies.get(d, dict())
            seq_acc[min_sequence] = mean(accuracies)
            all_accuracies[d] = seq_acc

            # print(f'\n\tAverage: {mean(accuracies):.3f}')
            # print("---------\n")

N = 30
tops = dict()
for i in range(1, MAX_SEQ):
    res = nlargest(N, all_accuracies, key=lambda x: all_accuracies[x][i])
    best = list()
    for model in res:
        best.append((model, all_accuracies[model][i]))
    tops[str(i)] = best

pp = pprint.PrettyPrinter(indent=4)

pp.pprint(tops)

with open("./best_models.json", 'w') as outfile:
    json.dump(tops, outfile)
