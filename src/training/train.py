# Partially inspired by https://github.com/jordiae/DeepLearning-MAI
import torch
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
from utils import load_architecture, evaluate
from training.covid_torch_dataset import CovidDataset, CovidData, DatasetBuilder
from training.models import FocalLoss
from data_analysis.extract_features import extract_features_information_metrics
from torch.nn.utils.rnn import pad_sequence
import os
import json
from importlib import reload
import numpy as np
from collections import defaultdict
from sklearn import metrics
import copy
from torchsampler import ImbalancedDatasetSampler  # (https://github.com/ufoym/imbalanced-dataset-sampler)
from collections import Counter
import random


def deterministic(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args: argparse.Namespace, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader,
          optimizer: torch.optim, criterion: torch.nn.Module, model: torch.nn.Module, device, exp_dir: str,
          checkpoint_suffix: str = '', metric: str = 'accuracy'):
    metric_args = {} if metric == 'accuracy' else {'average': 'macro'}
    metric_args_other = {} if not (metric == 'accuracy') else {'average': 'macro'}
    metric_score = metrics.accuracy_score if metric == 'accuracy' else metrics.f1_score
    metric_score_other = metrics.accuracy_score if not(metric == 'accuracy') else metrics.f1_score
    writer = SummaryWriter(log_dir=exp_dir)
    with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    logging.info(args)
    logging.info(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Training {params} parameters')
    epochs_without_improvement = 0
    best_valid_metric = -1
    best_valid_other_metric = -1
    best_loss_val = 1e6
    last_models_list = [None] * args.early_stop if args.early_stop != -1 else [None]
    t0 = time.time()
    last_output = None
    for epoch in range(1, args.epochs):
        # Train step
        model.train()
        logging.info(f'Epoch {epoch} |')
        loss_train = 0.0
        total = 0
        predictions_train = defaultdict(list)
        corrects_train = defaultdict(list)
        for idx, data in enumerate(train_loader):
            if (idx + 1) % 10 == 0:
                logging.info(f'{idx + 1}/{len(train_loader)} batches')
            patient_ids, static_data, dynamic_data, lengths, labels = data[0], data[1].to(device), \
                                                                      data[2], data[3].to(device), \
                                                                      data[4].to(device)
            if args.max_seq != -1:
                new_dynamic_data = []
                for data in dynamic_data:
                    new_dynamic_data.append(data[len(data) - args.max_seq:] if len(data) > args.max_seq else data)
                dynamic_data = new_dynamic_data

            dynamic_data = [data.to(device) for data in dynamic_data]  # move to cuda each element of the list
            dynamic_data = pad_sequence(dynamic_data, batch_first=True, padding_value=0).to(device)
            model.zero_grad()

            loss = 0.0
            effective_lengths = torch.ones(dynamic_data.shape[0]).long().to(device)
            c_lengths = torch.tensor(list(range(dynamic_data.shape[1]))).long().to(device)
            outputs = torch.zeros(dynamic_data.shape[0]).to(device)
            hidden = model.init_hidden(dynamic_data.shape[0])
            max_seq_len = dynamic_data.shape[1]

            dynamic_data_history = \
                torch.zeros(dynamic_data.shape[0], dynamic_data.shape[1], args.hidden_size).to(device)
            for seq_step in range(max_seq_len):
                events = dynamic_data[:, seq_step, :]
                non_zero = (effective_lengths != 0).nonzero().squeeze()
                static = static_data[non_zero]
                lens = effective_lengths[non_zero]
                evs = events[non_zero]
                if len(lens.shape) != 1:
                    static = static.unsqueeze(dim=0)
                    lens = lens.unsqueeze(dim=0)
                    evs = evs.unsqueeze(dim=0)
                evs = evs.unsqueeze(dim=1)
                if args.arch != 'lstm':
                    if len(non_zero.shape) == 0:
                        outputs[non_zero], hidden[:, non_zero:non_zero + 1, :], dynamic_data_event, _, _ = \
                            model((static, evs, lens, hidden, dynamic_data_history), seq_step)
                    else:
                        outputs[non_zero], hidden[:, non_zero, :], dynamic_data_event, _, _ = \
                            model((static, evs, lens, hidden, dynamic_data_history), seq_step)
                else:
                    outputs[non_zero], h, dynamic_data_event, _, _ = model(
                        (static, evs, lens, hidden, dynamic_data_history), seq_step)
                    if len(non_zero.shape) == 0:
                        hidden[0][:, non_zero:non_zero + 1, :] = h[0]
                        hidden[1][:, non_zero:non_zero + 1, :] = h[1]
                    else:
                        hidden[0][:, non_zero, :] = h[0]
                        hidden[1][:, non_zero, :] = h[1]

                dynamic_data_history[:, seq_step, :] = dynamic_data_event

                total += 1 if len(non_zero.shape) == 0 else len(non_zero)
                predicted = (torch.sigmoid(outputs[non_zero]).clone().data >= args.logistic_threshold).long()
                predicted = predicted.tolist()
                predicted = predicted if isinstance(predicted, list) else [predicted]
                for pred in predicted:
                    predictions_train[seq_step].append(pred)

                trues = labels[non_zero].clone().data.tolist()
                trues = trues if isinstance(trues, list) else [trues]
                for true in trues:
                    corrects_train[seq_step].append(true)
                if outputs[non_zero].size():
                    if args.criterion == 'bce_logits':
                        loss += criterion(outputs[non_zero].clone(), labels[non_zero].float())
                    else:
                        loss += criterion(torch.sigmoid(outputs[non_zero]).clone(), labels[non_zero].float())
                effective_lengths = (c_lengths[seq_step] < lengths - 1).long()
            if hasattr(loss, 'backward'):
                loss.backward()

            if args.clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping)

            if hasattr(loss, 'item'):
                optimizer.step()
                loss_train += loss.item()

        # compute the weighted average, that is equivalent to
        # the total number of correct predictions over the total number of examples
        accuracy_train_steps = [metric_score(corrects_train[step], predictions_train[step], **metric_args)
                                for step in range(len(predictions_train.keys()))]
        accuracy_train_weigths = [len(predictions_train[step]) for step in range(len(predictions_train.keys()))]
        accuracy_train = np.average(accuracy_train_steps, weights=accuracy_train_weigths)

        loss_train_avg = loss_train / total
        logging.info(f'train: avg_loss = {loss_train_avg:.5f} | {metric} = {accuracy_train:.3f}')
        writer.add_scalar('Avg-loss/train', loss_train_avg, epoch + 1)
        writer.add_scalar(f'{"Accuracy" if metric == "accuracy" else metric}/train', accuracy_train, epoch + 1)

        # Valid step
        total = 0
        loss_val = 0
        model.eval()
        with torch.no_grad():
            predictions_valid = defaultdict(list)
            corrects_valid = defaultdict(list)
            for data in valid_loader:
                patient_ids, static_data, dynamic_data, lengths, labels = data[0], data[1].to(device), \
                                                                          data[2], data[3].to(device), \
                                                                          data[4].to(device)
                dynamic_data = pad_sequence(dynamic_data, batch_first=True, padding_value=0).to(device)

                effective_lengths = torch.ones(dynamic_data.shape[0]).long().to(device)
                c_lengths = torch.tensor(list(range(dynamic_data.shape[1]))).long().to(device)
                outputs = torch.zeros(dynamic_data.shape[0]).to(device)
                hidden = model.init_hidden(dynamic_data.shape[0])
                max_seq_len = dynamic_data.shape[1]
                dynamic_data_history = torch.zeros(len(data[0]), dynamic_data.shape[1], args.hidden_size).to(device)
                for seq_step in range(max_seq_len):
                    events = dynamic_data[:, seq_step, :]
                    non_zero = (effective_lengths != 0).nonzero().squeeze()
                    static = static_data[non_zero]
                    lens = effective_lengths[non_zero]
                    evs = events[non_zero]
                    if len(lens.shape) != 1:
                        static = static.unsqueeze(dim=0)
                        lens = lens.unsqueeze(dim=0)
                        evs = evs.unsqueeze(dim=0)
                    evs = evs.unsqueeze(dim=1)
                    if args.arch != 'lstm':
                        if len(non_zero.shape) == 0:
                            outputs[non_zero], hidden[:, non_zero:non_zero + 1, :], dynamic_data_event, _, _ = model(
                                (static, evs, lens, hidden, dynamic_data_history), seq_step)
                        else:
                            outputs[non_zero], hidden[:, non_zero, :], dynamic_data_event, _, _ = \
                                model((static, evs, lens, hidden, dynamic_data_history), seq_step)
                    else:
                        outputs[non_zero], h, dynamic_data_event, _, _ = \
                            model((static, evs, lens, hidden, dynamic_data_history), seq_step)
                        if len(non_zero.shape) == 0:
                            hidden[0][:, non_zero:non_zero + 1, :] = h[0]
                            hidden[1][:, non_zero:non_zero + 1, :] = h[1]
                        else:
                            hidden[0][:, non_zero, :] = h[0]
                            hidden[1][:, non_zero, :] = h[1]

                    dynamic_data_history[:, seq_step, :] = dynamic_data_event
                    dynamic_data_history.to(device)
                    total += 1 if len(non_zero.shape) == 0 else len(non_zero)
                    predicted = (torch.sigmoid(outputs[non_zero]).clone().data >= args.logistic_threshold).long()
                    predicted = predicted.tolist()
                    predicted = predicted if isinstance(predicted, list) else [predicted]
                    for pred in predicted:
                        predictions_valid[seq_step].append(pred)

                    trues = labels[non_zero].clone().data.tolist()
                    trues = trues if isinstance(trues, list) else [trues]
                    for true in trues:
                        corrects_valid[seq_step].append(true)
                    if outputs[non_zero].size():
                        if args.criterion == 'bce_logits':
                            loss_val += criterion(outputs[non_zero].clone(), labels[non_zero].float())
                        else:
                            loss_val += criterion(torch.sigmoid(outputs[non_zero]).clone(), labels[non_zero].float())
                    effective_lengths = (c_lengths[seq_step] < lengths - 1).long()

        # compute the weighted average, that is equivalent to
        # the total number of correct predictions over the total number of examples
        accuracy_valid_steps = [metric_score(corrects_valid[step], predictions_valid[step], **metric_args)
                                for step in range(len(predictions_valid.keys()))]
        accuracy_valid_weigths = [len(predictions_valid[step]) for step in range(len(predictions_valid.keys()))]
        accuracy_valid = np.average(accuracy_valid_steps, weights=accuracy_valid_weigths)

        all_corrects = []
        all_predicted = []
        for i in range(len(corrects_valid)):
            all_corrects.extend(corrects_valid[i])
            all_predicted.extend(predictions_valid[i])
        all_metric = metric_score(all_corrects, all_predicted, **metric_args)
        all_metric_other = metric_score_other(all_corrects, all_predicted, **metric_args_other)
        loss_val_avg = loss_val / total
        logging.info(f'valid: avg_loss = {loss_val_avg:.5f} | {metric} = {all_metric:.3f}')
        writer.add_scalar('Avg-loss/valid', loss_val_avg, epoch + 1)
        writer.add_scalar(f'{"Accuracy" if metric == "accuracy" else metric}/valid', all_metric, epoch + 1)
        checkpoint_last_name = 'checkpoint_last.pt' if len(checkpoint_suffix) == 0 else \
            f'checkpoint_last_{checkpoint_suffix}.pt'
        torch.save(model.state_dict(), os.path.join(exp_dir, checkpoint_last_name))
        eps = 0.001
        # if loss_val_avg < best_loss_val and best_loss_val - loss_val_avg >= eps:
        if all_metric > best_valid_metric:
            epochs_without_improvement = 0
            best_loss_val = loss_val_avg
            best_valid_metric = all_metric
            best_valid_other_metric = all_metric_other
            checkpoint_best_name = 'checkpoint_best.pt' if len(checkpoint_suffix) == 0 else \
                f'checkpoint_best_{checkpoint_suffix}.pt'
            torch.save(model.state_dict(), os.path.join(exp_dir, checkpoint_best_name))
            best_model = copy.deepcopy(model)
            logging.info(f'best valid loss: {best_loss_val:.3f}')
            logging.info(f'best valid {metric}: {best_valid_metric:.3f}')
        else:
            epochs_without_improvement += 1
            logging.info(f'best valid loss: {best_loss_val:.3f}')
            logging.info(f'best valid {metric}: {best_valid_metric:.3f}')
            if args.early_stop != -1 and epochs_without_improvement == args.early_stop:
                break
        last_models_list.pop()
        last_models_list.insert(0, copy.deepcopy(model))
        logging.info(f'{epochs_without_improvement} epochs without improvement in validation set')

    t1 = time.time()
    logging.info(f'Finished training in {t1 - t0:.1f}s')
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device('cuda')
    else:
        device = 'cpu'
    model = load_architecture(device, args)
    if torch.cuda.is_available() and not args.no_cuda:
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    model.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model.to(device)

    # Model visualization
    if args.visualize:
        from torchviz import make_dot
        make_dot(last_output).render(os.path.join(exp_dir, "network"))

    extra_name_pickle = f'_{checkpoint_suffix}' if len(checkpoint_suffix) != 0 else ''
    logging.info('VALIDATION')
    _, _ = evaluate(valid_loader, [model], device, 'valid' + extra_name_pickle, criterion,
                    args.logistic_threshold, exp_dir, metric=metric, max_seq=args.max_seq)
    logging.info('TEST')
    eval_loss, eval_metrics = evaluate(test_loader, [model], device, 'test' + extra_name_pickle,
                                       criterion, args.logistic_threshold, exp_dir, metric=metric,
                                       max_seq=args.max_seq)
    metric_other = 'f1' if metric == 'accuracy' else 'accuracy'
    return {f'validation_{metric}': best_valid_metric, f'validation_{metric_other}': best_valid_other_metric,
            'validation_loss': float(loss_val),
            f'evaluation_{metric}': eval_metrics[f'{metric}_avg_weighted'], 'evaluation_loss': eval_loss,
            'evaluation_metrics': eval_metrics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help='Experiment name')
    parser.add_argument('data_path', type=str, help='Path to data')
    parser.add_argument('--use-attention', action='store_true', help='Use attention mechanisms for classification')
    parser.add_argument('--attention-type', type=str, help='Attention type to be used', default='dot')
    parser.add_argument('--attention-fields', type=str, help='Fields to use attention with: both or static_dynamic',
                        default='both')
    parser.add_argument('--arch', type=str, help='Architecture', default='lstm')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=200)
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001)
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--optimizer', type=str, help='Optimizer', default='Adam')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=32)
    parser.add_argument('--criterion', type=str, help='Possible options, "bce", "bce_loss" or "focal_loss"',
                        default='bce')
    parser.add_argument('--focal-loss-gamma', type=float, help='Gamma coefficient of focal loss', default=2)
    parser.add_argument('--use-prior-prob-label', action='store_true',
                        help='Set a prior probability for the positive label')
    parser.add_argument('--early-stop', type=int,
                        help='Patience in early stop in validation set (-1 -> no early stop)', default=10)
    parser.add_argument('--weight-decay', type=float, help='Weight decay', default=0.0001)
    parser.add_argument('--dropout', type=float, help='Dropout in RNN and FC layers', default=0.15)
    parser.add_argument('--static-input-size', type=int, help='Size of the static input')
    parser.add_argument('--dynamic-input-size', type=int, help='Size of the dynamic input')
    parser.add_argument('--static-embedding-size', type=int, help='Size of the static embedding', default=64)
    parser.add_argument('--dynamic-embedding-size', type=int, help='Size of the dynamic embedding', default=64)
    parser.add_argument('--hidden-size', type=int, help='Hidden state size of the RNN', default=128)
    parser.add_argument('--rnn-layers', type=int, help='Number of recurrent layers', default=1)
    parser.add_argument('--fc-layers', type=int, help='Number of fully-connected layers after the RNN-static concat',
                        default=2)
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional RNN in the encoder')
    parser.add_argument('--clipping', type=float, help='Gradient clipping', default=0.25)
    parser.add_argument('--train-seed', type=int, help='Random seed for training', default=42)
    parser.add_argument('--param-search-config', type=str, help='Configuration file with hyperparameters grid search')
    parser.add_argument('--param-search-resume', type=int, help='Resume param search from an index', default=0)
    parser.add_argument('--logistic-threshold', type=float, help='Threshold of the logistic regression (default 0.5)',
                        default=0.5)
    parser.add_argument('--use-imbalanced', action='store_true', help='Use imbalanced dataset for training')
    parser.add_argument('--visualize', action='store_true', help='Visualize network graph')
    parser.add_argument('--no-stratified', action='store_true', help='Disables stratified split')
    parser.add_argument('--cross-val', type=int, help='K-Fold cross validation (if 1, ignored)', default=1)
    parser.add_argument('--nested-cross-val', type=int, help='Nested K-Fold cross validation (if 1, ignored)',
                        default=1)
    parser.add_argument('--remove-outliers-file', type=str,
                        help='File with a list of patient ids to filter out that have been identified as outliers'
                             'with a certain outlier detection criteria')
    parser.add_argument('--features-file', type=str,
                        help='File with a list of features ranked with information metrics')
    parser.add_argument('--features-top-n', type=int, help='Top N common features to select')
    parser.add_argument('--metric', type=str, choices=['accuracy', 'f1'], help='Metric to optimize and display',
                        default='accuracy')
    parser.add_argument('--max-seq', type=int, help='Maximum sequence length (longer sequences are truncated from the'
                                                    'end default: -1 -> do not truncate)', default=-1)
    # parser.add_argument('--length-dropout')
    args = parser.parse_args()

    device = torch.device('cuda:0' if not args.no_cuda and torch.cuda.is_available() else 'cpu')

    deterministic(args.train_seed)

    # Read patient ids of the outliers
    filter_pids = []
    if args.remove_outliers_file:
        with open(args.remove_outliers_file) as fn:
            outliers_pids = list(json.load(fn).keys())
            filter_pids.extend(outliers_pids)
    logging.basicConfig(filename='', level=logging.INFO)

    logging.info(f'Filtered out {(len(filter_pids) / 2307) * 100}  % of pids')

    if args.param_search_config:
        with open(args.param_search_config) as fn:
            grid_params = json.load(fn)

        # train loop over the hypeparameters combinations
        items = sorted(grid_params.items())
        keys, values = zip(*items)

        for i in range(args.param_search_resume, len(values[0])):
            logging.shutdown()
            reload(logging)
            params = dict(zip(keys, [v[i] for v in values]))
            args.__dict__.update(params)

            # Read features and select the top-ranked ones
            features = []
            if args.features_file:
                if args.features_top_n:
                    features = extract_features_information_metrics(args.features_file, args.features_top_n)
                else:
                    features = extract_features_information_metrics(args.features_file)
            data = CovidData(args.data_path, 'dataset_static.csv', 'dataset_dynamic.csv', 'dataset_labels.csv',
                             filter_patients_ids=filter_pids,
                             features_selected=features)

            train_dataset = CovidDataset(data=data, subset='train', stratified=not args.no_stratified,
                                         seed=args.train_seed)
            valid_dataset = CovidDataset(data=data, subset='valid', stratified=not args.no_stratified,
                                         seed=args.train_seed)
            test_dataset = CovidDataset(data=data, subset='test', stratified=not args.no_stratified,
                                        seed=args.train_seed)

            if args.use_imbalanced:
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                          collate_fn=CovidDataset.collate,
                                          sampler=ImbalancedDatasetSampler(train_dataset,
                                                                           callback_get_label=train_dataset.get_label))
            else:
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=CovidDataset.collate)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=CovidDataset.collate)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=CovidDataset.collate)

            # if not provided, set automatically set the static and dynamic input sizes
            args.static_input_size = list(train_loader)[0][1].shape[-1]
            args.dynamic_input_size = list(train_loader)[0][2][0].shape[-1]
            model = load_architecture(device, args)

            if args.optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                            momentum=0.9)
            else:
                raise NotImplementedError('Optimizer not implemented')

            if args.criterion == 'bce':
                criterion = torch.nn.BCELoss()
            elif args.criterion == 'bce_logits':
                label_count = Counter(train_dataset.labels)
                pos_weight = torch.Tensor([label_count[0] / label_count[1]]).to(device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            elif args.criterion == 'focal_loss':
                criterion = FocalLoss(args.focal_loss_gamma)
            else:
                raise NotImplementedError('Criterion not implemented')

            timestamp = time.strftime("%Y-%m-%d-%H%M")
            # params_string = params.__str__().replace(': ', '-').replace("'", '').replace(', ', '_')
            exp_dir = os.path.join('..', 'experiments', f'{args.experiment_name}-search-{i}-{timestamp}')
            os.makedirs(exp_dir, exist_ok=True)

            with open(os.path.join(exp_dir, 'train_patient_ids.json'), 'w') as jsfile:
                json.dump({"train": list(map(float, train_loader.dataset.patient_id)),
                           "validation": list(map(float, valid_loader.dataset.patient_id)),
                           "test": list(map(float, test_loader.dataset.patient_id))}, jsfile)

            log_path = os.path.join(exp_dir, 'train.log')
            logging.basicConfig(filename=log_path, level=logging.INFO)
            logging.getLogger('').addHandler(logging.StreamHandler())
            logging.info(f'Experiment {i} with params: {params}')
            logging.info(params)
            try:
                if args.cross_val != 1 or args.nested_cross_val != 1:
                    raise NotImplementedError('Cross-validation is deprecated in train.py. Use selection_cv.py')
                train(args, train_loader, valid_loader, test_loader, optimizer, criterion, model, device, exp_dir,
                      metric=args.metric)
            except Exception as e:
                logging.info(f'Failed execution with parameters: {params}')
                logging.info(str(e))
                continue
    else:
        # Read features and select the top-ranked ones
        features = []
        if args.features_file:
            if args.features_top_n:
                features = extract_features_information_metrics(args.features_file, args.features_top_n)
            else:
                features = extract_features_information_metrics(args.features_file)
        data = CovidData(args.data_path, 'dataset_static.csv', 'dataset_dynamic.csv', 'dataset_labels.csv',
                         filter_patients_ids=filter_pids,
                         features_selected=features)
        train_dataset = CovidDataset(data=data, subset='train', stratified=not args.no_stratified,
                                     seed=args.train_seed)
        valid_dataset = CovidDataset(data=data, subset='valid', stratified=not args.no_stratified,
                                     seed=args.train_seed)
        test_dataset = CovidDataset(data=data, subset='test', stratified=not args.no_stratified,
                                    seed=args.train_seed)
        if args.use_imbalanced:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      collate_fn=CovidDataset.collate,
                                      sampler=ImbalancedDatasetSampler(train_dataset,
                                                                       callback_get_label=train_dataset.get_label))
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=CovidDataset.collate)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=CovidDataset.collate)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=CovidDataset.collate)

        device = torch.device('cuda:0' if not args.no_cuda and torch.cuda.is_available() else 'cpu')

        # if not provided, set automatically set the static and dynamic input sizes
        args.static_input_size = args.static_input_size if args.static_input_size else \
            list(train_loader)[0][1].shape[-1]
        args.dynamic_input_size = args.dynamic_input_size if args.dynamic_input_size else \
            list(train_loader)[0][2][0].shape[-1]
        model = load_architecture(device, args)

        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                        momentum=0.9)
        else:
            raise NotImplementedError('Optimizer not implemented')

        if args.criterion == 'bce':
            criterion = torch.nn.BCELoss()
        elif args.criterion == 'bce_logits':
            label_count = Counter(train_dataset.labels)
            pos_weight = torch.Tensor([label_count[0] / label_count[1]]).to(device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif args.criterion == 'focal_loss':
            criterion = FocalLoss(args.focal_loss_gamma)
        else:
            raise NotImplementedError('Criterion not implemented')

        timestamp = time.strftime("%Y-%m-%d-%H%M")
        exp_dir = os.path.join('..', 'experiments', f'{args.experiment_name}-{timestamp}')
        os.makedirs(exp_dir, exist_ok=True)

        log_path = os.path.join(exp_dir, 'train.log')
        logging.basicConfig(filename=log_path, level=logging.INFO)
        logging.getLogger('').addHandler(logging.StreamHandler())

        with open(os.path.join(exp_dir, 'train_patient_ids.json'), 'w') as jsfile:
            json.dump({"train": list(map(float, train_loader.dataset.patient_id)),
                       "validation": list(map(float, valid_loader.dataset.patient_id)),
                       "test": list(map(float, test_loader.dataset.patient_id))}, jsfile)

        if args.cross_val != 1 or args.nested_cross_val != 1:
            raise NotImplementedError('Cross-validation is deprecated in train.py. Use selection_cv.py')
        train(args, train_loader, valid_loader, test_loader, optimizer, criterion, model, device, exp_dir,
              metric=args.metric)
