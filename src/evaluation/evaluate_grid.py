import argparse
from .utils import load_architecture, evaluate, ArgsStruct
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


def deterministic(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('models_path', type=str, help='Path to model directory.')
    parser.add_argument('data_path', type=str, help='Path to data')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=32)
    parser.add_argument('--criterion', type=str, help='Criterion', default='bce')
    parser.add_argument('--subset', type=str, help='Subset to evaluate on (train, valid, test)', default='test')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--data-seed', type=int, help='Random seed for data', default=42)
    parser.add_argument('--eval-seed', type=int, help='Random seed for evaluation', default=42)
    parser.add_argument('--logistic-threshold', type=float, help='Threshold of the logistic regression (default 0.5)',
                        default=0.5)
    parser.add_argument('--no-stratified', action='store_true', help='Disables stratified split')
    parser.add_argument('--top-n-models', type=int, help='Top N models', default=1)
    args_eval = parser.parse_args()

    deterministic(args_eval.eval_seed)

    timestamp = time.strftime("%Y-%m-%d-%H%M")

    log_path = os.path.join(f'eval-{args_eval.subset}--evaluation-{timestamp}.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    logging.info(args_eval)

    device = torch.device('cuda:0' if not args_eval.no_cuda and torch.cuda.is_available() else 'cpu')

    data = CovidData(args_eval.data_path, 'dataset_static.csv', 'dataset_dynamic.csv',
                     'dataset_labels.csv')

    dataset = CovidDataset(data=data, subset=args_eval.subset, stratified=not args_eval.no_stratified,
                           seed=args_eval.data_seed)

    data_loader = DataLoader(dataset, batch_size=args_eval.batch_size, shuffle=False, collate_fn=CovidDataset.collate)

    if args_eval.criterion == 'bce':
        criterion = torch.nn.BCELoss()
    else:
        raise NotImplementedError('Criterion not implemented')
    base_path = args_eval.models_path
    args_eval.models_path = [os.path.join(args_eval.models_path, path) for path in os.listdir(args_eval.models_path)
                             if os.path.isdir(os.path.join(args_eval.models_path, path))]

    models = []
    for model_path in args_eval.models_path:
        with open(os.path.join(model_path, 'args.json'), 'r') as f:
            train_args = ArgsStruct(**json.load(f))
        args = train_args

        device = 'cpu'
        model = load_architecture(device, args)
        if torch.cuda.is_available() and not args.no_cuda:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint_best.pt'), map_location=map_location))
        model.to(device)

        models.append(model)

    evaluations = list()

    for model_idx, model in enumerate(models):
        timestamp = time.strftime("%Y-%m-%d-%H%M")
        exp_dir = os.path.join('..', 'evaluation', f'{model_idx}-{timestamp}')
        os.makedirs(exp_dir, exist_ok=True)
        evaluations.append(evaluate(data_loader, [model], device, args_eval.subset, criterion, args_eval.eval_seed,
                 args_eval.logistic_threshold, exp_dir=exp_dir))

    if len(evaluations) > 1:
        if type(evaluations[0]) == list:
            indexes = np.argsort([mean([evaluation2[1] for evaluation2 in evaluation])
                                  for evaluation in evaluations])[:min(len(evaluations), args_eval.top_n_models)]
        else:
            # Search for highest accuracy
            indexes = np.argsort([evaluation[1] for evaluation in evaluations])[:min(len(evaluations),
                                                                                     args_eval.top_n_models)]
        for idx, i in enumerate(indexes):
            print(idx, "model", models[i], "evaluations:", "loss:", evaluations[i][0], "accuracy:", evaluations[i][1])
        print("Best model paths are")
        best_models = list()
        for idx, i, in enumerate(indexes):
            best_models.append(args_eval.models_path[i])
            print(idx, args_eval.models_path[i])
        with open(os.path.join(base_path, 'best_models.json'), 'w') as outfile:
            json.dump(best_models, outfile)

    else:
        with open(os.path.join(base_path, 'best_models.json'), 'w') as outfile:
            json.dump(args_eval.models_path[0], outfile)
