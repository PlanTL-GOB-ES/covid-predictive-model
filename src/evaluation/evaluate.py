import argparse
from utils import load_architecture, evaluate, ArgsStruct
import torch
import os
import json
from training.covid_torch_dataset import CovidDataset, CovidData, DatasetBuilder
from torch.utils.data.dataloader import DataLoader
import logging
import shutil
import time
import numpy as np
from statistics import mean, stdev
from typing import List


def deterministic(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def cross_validate(cv: int, nested_cv: int, args_eval: argparse.Namespace, data: CovidData, criterion: torch.nn.Module,
                   model: torch.nn.Module, device, exp_dir):
    db = DatasetBuilder(data=data)
    # Outer loop
    scores = []
    for i, (inner_train_valids, test_dataset) in enumerate(db.build_datasets(cv=cv, nested_cv=nested_cv)):
        if args_eval.subset == 'test':
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=CovidDataset.collate)
            eval_loss, eval_acc = evaluate(test_loader, [model], device, args_eval.subset, criterion,
                                           args_eval.logistic_threshold, exp_dir, max_seq=args_eval.max_seq)
            scores.append((eval_loss, eval_acc))
        else:
            for j, (train_dataset, valid_dataset) in enumerate(inner_train_valids):
                if args_eval.subset == 'train':
                    data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                             collate_fn=CovidDataset.collate)
                else:
                    data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                             collate_fn=CovidDataset.collate)
                eval_loss, eval_acc = evaluate(data_loader, [model], device, args_eval.subset + f'-{i}-{j}',
                                               criterion, args_eval.logistic_threshold, exp_dir,
                                               max_seq=args_eval.max_seq)
                scores.append((eval_loss, eval_acc))
    logging.info(f'-{i}-{j} Cross-validation. Subset {args_eval.subset}')
    for idx, (eval_loss, eval_acc) in enumerate(scores):
        logging.info(f'{idx}: eval_loss = {eval_loss} | eval_acc = {eval_acc}')

    def aggregate(metric):
        return dict(
                    max=max(metric),
                    min=min(metric),
                    mean=mean(metric),
                    std=stdev(metric)
                )
    res = {}
    for _, metrics in scores:
        for e in metrics:
            if e not in res:
                res[e] = [metrics[e]]
            else:
                res[e].append(metrics[e])
    for e in res:
        res[e] = aggregate(res[e])
    logging.info('Metrics: ' + str(res))


def cross_validate_ensemble(cv: int, nested_cv: int, args_eval: argparse.Namespace, data: CovidData,
                            criterion: torch.nn.Module, folds: List[List[torch.nn.Module]], device, exp_dir):
    db = DatasetBuilder(data=data, seed=42, seed_cv=15)
    # Outer loop
    scores = []
    for i, (inner_train_valids, test_dataset) in enumerate(db.build_datasets(cv=cv, nested_cv=nested_cv)):
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=CovidDataset.collate)
        if args_eval.subset == 'test':
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=CovidDataset.collate)
            eval_loss, eval_acc = evaluate(test_loader, folds[i], device, args_eval.subset, criterion,
                                           args_eval.logistic_threshold, exp_dir, max_seq=args_eval.max_seq,
                                           aggregate=args_eval.aggregate, aggregate_or_th=args_eval.aggregate_or_th)
            scores.append((eval_loss, eval_acc))
        else:
            for j, (train_dataset, valid_dataset) in enumerate(inner_train_valids):
                if args_eval.subset != 'cv-test':
                    if args_eval.subset == 'train':
                        data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                 collate_fn=CovidDataset.collate)
                    else:
                        data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                                 collate_fn=CovidDataset.collate)
                    eval_loss, eval_acc = evaluate(data_loader, folds[i], device, args_eval.subset + f'-{i}-{j}',
                                                   criterion, args_eval.logistic_threshold, exp_dir,
                                                   max_seq=args_eval.max_seq,
                                                   aggregate=args_eval.aggregate,
                                                   aggregate_or_th=args_eval.aggregate_or_th)
                else:
                    eval_loss, eval_acc = evaluate(test_loader, folds[j], device, args_eval.subset + f'-{i}-{j}',
                                                   criterion, args_eval.logistic_threshold, exp_dir,
                                                   max_seq=args_eval.max_seq,
                                                   aggregate=args_eval.aggregate,
                                                   aggregate_or_th=args_eval.aggregate_or_th)
                scores.append((eval_loss, eval_acc))
    if args_eval.subset != 'test':
        logging.info(f'-{i}-{j} Cross-validation. Subset {args_eval.subset}')
    else:
        logging.info(f'{i} Cross-validation. Subset {args_eval.subset}')

    for idx, (eval_loss, eval_acc) in enumerate(scores):
        logging.info(f'{idx}: eval_loss = {eval_loss} | eval_acc = {eval_acc}')

    def aggregate(metric):
        return dict(
                    max=max(metric),
                    min=min(metric),
                    mean=mean(metric),
                    std=stdev(metric)
                )
    res = {}
    for _, metrics in scores:
        for e in metrics:
            if e not in res:
                res[e] = [metrics[e]]
            else:
                res[e].append(metrics[e])
    for e in res:
        res[e] = aggregate(res[e])
    logging.info('Metrics: ' + str(res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('models_path', type=str, help='Path to model directory. If more than one path is provided, an'
                                                      'ensemble of models is loaded', nargs='+')
    parser.add_argument('data_path', type=str, help='Path to data')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=32)
    parser.add_argument('--criterion', type=str, help='Criterion', default='bce')
    parser.add_argument('--subset', type=str, help='Subset to evaluate on (train, valid, test, cv-test)',
                        default='cv-test')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--logistic-threshold', type=float, help='Threshold of the logistic regression (default 0.5)',
                        default=0.5)
    parser.add_argument('--no-stratified', action='store_true', help='Disables stratified split')
    parser.add_argument('--cross-val', type=int, help='K-Fold cross validation', default=-1)
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
    args_eval = parser.parse_args()

    deterministic(args_eval.seed)

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    path = os.path.join('..', 'evaluations', f'eval-{timestamp}')
    os.makedirs(path, exist_ok=True)
    # extra = ''
    # if len(args_eval.models_path) > 1:
    #    extra = '-ensemble'
    # elif args_eval.cross_val != -1 or args_eval.nested_cross_val != -1:
    #    extra = f'{args_eval.nested_cross_val}x{args_eval.cross_val}'
    # log_path = os.path.join('..', 'evaluations', f'eval-{args_eval.subset + extra}-{timestamp}.log')
    log_path = os.path.join(path, 'eval.log')

    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    logging.info(args_eval)

    device = torch.device('cuda:0' if not args_eval.no_cuda and torch.cuda.is_available() else 'cpu')

    data = CovidData(args_eval.data_path, 'dataset_static.csv', 'dataset_dynamic.csv',
                     'dataset_labels.csv')

    dataset = CovidDataset(data=data, subset=args_eval.subset, stratified=not args_eval.no_stratified,
                           seed=args_eval.seed)

    data_loader = DataLoader(dataset, batch_size=args_eval.batch_size, shuffle=False, collate_fn=CovidDataset.collate)

    if args_eval.criterion == 'bce':
        criterion = torch.nn.BCELoss()
    else:
        raise NotImplementedError('Criterion not implemented')

    if args_eval.cross_val == -1 and args_eval.nested_cross_val == -1:
        logging.warning('The actual values used for cross-validation are the ones used in training.')

    if args_eval.cross_val == -1 and args_eval.nested_cross_val == -1:
        models = []
        for model_path in args_eval.models_path:
            with open(os.path.join(model_path, 'args.json'), 'r') as f:
                train_args = ArgsStruct(**json.load(f))
            args = train_args
            model = load_architecture(device, args)
            if torch.cuda.is_available() and not args.no_cuda:
                map_location = torch.device('cuda:0')
            else:
                map_location = torch.device('cpu')
            model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint_best.pt'), map_location=map_location))
            model.to(device)
            models.append(model)
        evaluate(data_loader, models, device, criterion, args_eval.seed,
                 args_eval.logistic_threshold, exp_dir=path, max_seq=args_eval.max_seq)
        #for model_path in args_eval.models_path:
        #    shutil.copy(log_path, os.path.join(model_path, log_path))
        #os.remove(log_path)
    elif args_eval.nested_cross_val == -1 and args_eval.cross_val != -1 and len(args_eval.models_path) > 1:
        folds = [[] for _ in range(args_eval.cross_val)]
        for idx, model_path in enumerate(args_eval.models_path):
            #models = []
            with open(os.path.join(model_path, 'args.json'), 'r') as f:
                train_args = ArgsStruct(**json.load(f))
            args = train_args
            for i in range(args_eval.cross_val):
                model = load_architecture(device, args)
                if torch.cuda.is_available() and not args.no_cuda:
                    map_location = torch.device('cuda:0')
                else:
                    map_location = torch.device('cpu')
                print('Loading', os.path.join(model_path, f'checkpoint_best_{i}.pt'))
                model.load_state_dict(torch.load(os.path.join(model_path, f'checkpoint_best_{i}.pt'),
                                                 map_location=map_location),
                                      )
                model.to(device)
                #models.append(model)
                folds[i].append(model)
            #folds[idx].append(models)
        cross_validate_ensemble(args.cross_val, args.nested_cross_val, args_eval, data, criterion, folds, device,
                                exp_dir=path)
        #for model_path in args_eval.models_path:
        #    shutil.copy(log_path, os.path.join(model_path, log_path))
        #os.remove(log_path)
    elif args_eval.nested_cross_val == -1 and args_eval.cross_val != -1 and len(args_eval.models_path) == 1:
        model_path = args_eval.models_path[0]
        with open(os.path.join(model_path, 'args.json'), 'r') as f:
            train_args = ArgsStruct(**json.load(f))
        args = train_args
        model = load_architecture(device, args)
        if torch.cuda.is_available() and not args.no_cuda:
            map_location = torch.device('cuda:0')
        else:
            map_location = torch.device('cpu')
        model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint_best.pt'), map_location=map_location))
        model.to(device)
        cross_validate(args.cross_val, args.nested_cross_val, args_eval, data, criterion, model, device, exp_dir=path)
        #shutil.copy(log_path, os.path.join(args.model_path[0], log_path))
        #os.remove(log_path)
    else:
        raise RuntimeError("Can't run ensemble with nested cross-validation")
