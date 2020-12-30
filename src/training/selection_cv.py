import argparse
from utils import load_architecture, evaluate, ArgsStruct
import torch
import os
import json
from collections import defaultdict
from covid_torch_dataset import CovidDataset, CovidData, DatasetBuilder
from torch.utils.data.dataloader import DataLoader
from train import train
import logging
import shutil
import time
import numpy as np
from statistics import mean, stdev
from collections import Counter
import pandas as pd
from data_analysis.extract_features import extract_features_information_metrics
from training.models import FocalLoss
from copy import deepcopy


def deterministic(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def cross_validate(cv: int, nested_cv: int, args: argparse.Namespace, data: CovidData, optimizer: torch.optim,
                   criterion: torch.nn.Module, model: torch.nn.Module, device, exp_dir: str, seed: int = 42,
                   seed_cv: int = 15, metric: str = 'accuracy'):
    assert metric in ['accuracy', 'f1']
    db = DatasetBuilder(data=data, seed=seed, seed_cv=seed_cv)
    # Outer loop
    scores = []
    patientids = []
    initialized_parameters = deepcopy(model.state_dict())
    for i, (inner_train_valids, test_dataset) in enumerate(db.build_datasets(cv=cv, nested_cv=nested_cv)):
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=CovidDataset.collate)
        for j, (train_dataset, valid_dataset) in enumerate(inner_train_valids):
            if args.criterion == 'bce_logits':
                label_count = Counter(train_dataset.labels)
                pos_weight = torch.Tensor([label_count[0] / label_count[1]]).to(device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=CovidDataset.collate)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=CovidDataset.collate)
            patientids.append({"train": list(map(float, train_loader.dataset.patient_id)),
                               "validation": list(map(float, valid_loader.dataset.patient_id)),
                               "test": list(map(float, test_loader.dataset.patient_id))})

            if nested_cv > 1 and cv > 1:
                checkpoint_suffix = f'{i}-{j}'
            elif nested_cv > 1:
                checkpoint_suffix = str(i)
            elif cv > 1:
                checkpoint_suffix = str(j)
            else:
                checkpoint_suffix = ''
            model.load_state_dict(deepcopy(initialized_parameters))
            scores.append(train(args, train_loader, valid_loader, test_loader, optimizer, criterion, model, device,
                                exp_dir, checkpoint_suffix=checkpoint_suffix, metric=metric))
    pd.DataFrame([score['evaluation_metrics'] for score in scores]).to_csv(os.path.join(exp_dir, 'scores.csv'))
    with open(os.path.join(exp_dir, "patient_ids_cv.json"), 'w') as patient_file:
        json.dump(patientids, patient_file)
    return [score['validation_accuracy'] for score in scores], [score['validation_f1'] for score in scores]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate models')
    parser.add_argument('best_path', type=str, help='Path to the file that selects best models inside the path.')
    parser.add_argument('models_path', type=str, help='Path to best models path.')
    parser.add_argument('data_path', type=str, help='Path to data')
    parser.add_argument('--early-stop', type=int,
                        help='Patience in early stop in validation set (-1 -> no early stop)', default=10)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=200)
    parser.add_argument('--optimizer', type=str, help='Optimizer', default='Adam')
    parser.add_argument('--subset', type=str, help='Subset to evaluate on (train, valid, test)', default='test')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--train-seed', type=int, help='Random seed for training', default=42)
    parser.add_argument('--train-seed-cv', type=int, help='Random seed for inner loop of CV', default=15)
    parser.add_argument('--logistic-threshold', type=float, help='Threshold of the logistic regression (default 0.5)',
                        default=0.5)
    parser.add_argument('--cross-val', type=int, help='K-Fold cross validation', default=1)
    parser.add_argument('--nested-cross-val', type=int, help='Nested K-Fold cross validation',
                        default=1)
    parser.add_argument('--no-stratified', action='store_true', help='Disables stratified split')
    parser.add_argument('--no-feature-selection', action='store_true', help='disables CUDA training')
    parser.add_argument('--top-n-models', type=int, help='Top N models', default=1)
    parser.add_argument('--metric', type=str, choices=['accuracy', 'f1'], help='Metric to optimize and display',
                        default='accuracy')
    parser.add_argument('--max-seq', type=int, help='Maximum sequence length (longer sequences are truncated from the'
                                                    'end default: -1 -> do not truncate)', default=-1)
    args_eval = parser.parse_args()
    base_path = args_eval.models_path

    with open(args_eval.best_path) as f:
        data = json.load(f)
    args_eval.models_path = [os.path.join(base_path, d) for d in data]
    deterministic(args_eval.train_seed)

    timestamp = time.strftime("%Y-%m-%d-%H%M")

    log_path = os.path.join(f'eval-{args_eval.subset}--evaluation-{timestamp}.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    logging.info(args_eval)

    device = torch.device('cuda:0' if not args_eval.no_cuda and torch.cuda.is_available() else 'cpu')

    data = CovidData(args_eval.data_path, 'dataset_static.csv', 'dataset_dynamic.csv',
                     'dataset_labels.csv')

    models = []
    train_args_list = []
    train_args_list_dict = []
    for model_path in args_eval.models_path:
        with open(os.path.join(model_path, 'args.json'), 'r') as f:
            loaded_args = json.load(f)
            train_args = ArgsStruct(**loaded_args)
            train_args_list_dict.append(loaded_args)
        args = train_args
        args.dynamic_input_size = data.dynamic_table.shape[1]-1
        args.static_input_size = data.static_data.shape[1]
        train_args_list.append(args)
        model = load_architecture(device, args)
        if torch.cuda.is_available() and not args.no_cuda:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = lambda storage, location: 'cpu'
        model.to(device)
        models.append(model)

    evaluations = defaultdict(list)
    for model_idx, model in enumerate(models):
        timestamp = time.strftime("%Y-%m-%d-%H%M")

        if args_eval.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=train_args_list[model_idx].lr,
                                         weight_decay=train_args_list[model_idx].weight_decay)
        else:
            raise NotImplementedError('Optimizer not implemented')
        args_dict = vars(args_eval).copy()
        args_dict.update(train_args_list_dict[model_idx])
        args_dict.update(vars(args_eval))
        args_dict = argparse.Namespace(**args_dict)
        args_dict.dynamic_input_size = data.dynamic_table.shape[1] - 1
        args_dict.static_input_size = data.static_data.shape[1]
        exp_dir = args_dict.models_path[model_idx]

        if args_dict.criterion == 'bce':
            criterion = torch.nn.BCELoss()
        elif args_dict.criterion == 'bce_logits':
            criterion = None
        elif args_dict.criterion == 'focal_loss':
            criterion = FocalLoss(args_dict.focal_loss_gamma)
        else:
            raise NotImplementedError('Criterion not implemented')

        features = []
        if args_dict.features_file and not args_eval.no_feature_selection:
            if args_dict.features_top_n:
                features = extract_features_information_metrics(args_dict.features_file, args_dict.features_top_n)
            else:
                features = extract_features_information_metrics(args_dict.features_file)
        filter_pids = []
        if args_dict.remove_outliers_file:
            with open(args_dict.remove_outliers_file) as fn:
                outliers_pids = list(json.load(fn).keys())
                filter_pids.extend(outliers_pids)
        data = CovidData(args_dict.data_path, 'dataset_static.csv', 'dataset_dynamic.csv', 'dataset_labels.csv',
                         filter_patients_ids=filter_pids,
                         features_selected=features)
        try:
            acc, f1 = cross_validate(args_eval.cross_val, args_eval.nested_cross_val, args_dict, data, optimizer,
                               criterion, model, device, exp_dir, args_eval.train_seed,
                               seed_cv=args_eval.train_seed_cv, metric=args_eval.metric)
            evaluations['acc'].append(acc)
            evaluations['f1'].append(f1)
        except BaseException as e:
            print(args_eval.models_path[model_idx], e)
            evaluations['acc'].append(None)
            evaluations['f1'].append(None)
            raise e

        #os.remove(log_path)

    for k, evaluations in evaluations.items():
        if len(evaluations) > 1:
            if type(evaluations[0]) == list:
                scores = [mean(evaluation) if evaluation is not None else 0.0 for evaluation in evaluations]
                indexes = np.argsort(scores)[::-1][:min(len(evaluations), args_eval.top_n_models)]

            else:
                # Search for highest accuracy
                scores = [evaluation[1] if evaluation is not None else 0.0 for evaluation in evaluations]
                indexes = np.argsort(scores)[::-1][:min(len(evaluations), args_eval.top_n_models)]
            for idx, i in enumerate(indexes):
                print(idx, "model", models[i], "validation:", f"{k}:", evaluations[i])
            print("Best model paths are")
            best_models = list()
            for idx, i, in enumerate(indexes):
                best_models.append((args_eval.models_path[i], scores[i]))
                print(idx, args_eval.models_path[i])
            with open(os.path.join(os.path.dirname(base_path), f'best_cv_models_{k}.json'), 'w') as outfile:
                json.dump(best_models, outfile)

        else:
            with open(os.path.join(os.path.dirname(base_path), f'best_cv_models_{k}.json'), 'w') as outfile:
                json.dump(args_eval.models_path[0], outfile)
