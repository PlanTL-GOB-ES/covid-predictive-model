import os
import argparse
import json
import pickle
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAMS = ["batch_size", "dropout", "static_embedding_size", "dynamic_embedding_size",
          "arch", "rnn_layers", "use_attention", "attention_fields",
          "features_top_n", "weight_decay", "criterion", "lr"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models_path', type=str, help='Models path')
    parser.add_argument('selection_path', type=str, help='Path where best selection is located (JSON ARRAY)')
    parser.add_argument('output', type=str, help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.selection_path) as f:
        best_dirs = json.load(f)

    votings_dict = dict()
    params_list = list()
    for root, dirs, _ in os.walk(args.models_path):
        for d in [d for d in dirs if d in best_dirs]:
            print(f'MODEL "{d}":\n')
            with open(os.path.join(root, d, 'args.json')) as f:
                data = json.load(f)
            data_params = dict()
            use_attention = False
            for param in PARAMS:
                if param == 'use_attention':
                    use_attention = data[param]
                if param == 'attention_fields' and use_attention or param != 'attention_fields':
                    votings_dict[param] = votings_dict.get(param, dict())
                    votings_dict[param][data[param]] = votings_dict[param].get(data[param], 0) + 1
                    data_params[param] = data[param]
                else:
                    data_params[param] = 'NA'

            # Search for accuracy
            fls = [file for file in os.listdir(os.path.join(root, d)) if "eval_preds_" in file]
            fls_dates = np.argsort([os.path.getmtime(os.path.join(root, d, file)) for file in fls])[::-1][0]
            file = os.path.join(root, d, fls[fls_dates])
            data = pickle.load(open(file, "rb"))

            predictions = dict()
            labels = dict()

            predicted_all = []
            for step in range(len(data['predictions_from_start'])):
                predicted_all.extend(data['predictions_from_start'][step])

            corrects_all = []
            for step in range(len(data['labels'])):
                corrects_all.extend(data['labels'][step])

            data_params['accuracy'] = metrics.accuracy_score(corrects_all, predicted_all)
            # End accuracy search
            params_list.append(data_params)

    print("Counts: ", votings_dict)
    for key, value in votings_dict.items():
        methods = list([method for method in value.keys()])
        x = list(range(0, len(methods)))
        y = list([val for val in value.values()])

        fig, ax = plt.subplots()
        plt.title(key)
        plt.bar(x, y)
        plt.xticks(x, methods)
        plt.savefig(os.path.join(args.output, key + '.png'))
        plt.show()

    pd.DataFrame(params_list).to_csv(os.path.join(args.output, 'best_params_list.csv'))
