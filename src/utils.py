import argparse
import logging
import os
import copy
from collections import defaultdict
from typing import List
import numpy as np
import torch
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from training.models import RNNClassifier
from operator import add
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


class ArgsStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_architecture(device: torch.device, args: argparse.Namespace):
    model = RNNClassifier(arch=args.arch, static_input_size=args.static_input_size,
                          dynamic_input_size=args.dynamic_input_size,
                          static_embedding_size=args.static_embedding_size,
                          hidden_size=args.hidden_size, dropout=args.dropout, rnn_layers=args.rnn_layers,
                          bidirectional=args.bidirectional, use_attention=args.use_attention,
                          attention_type=args.attention_type, attention_fields=args.attention_fields,
                          device=device, fc_layers=args.fc_layers, use_prior_prob_label=args.use_prior_prob_label)
    model.to(device)
    return model


def deterministic(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def evaluate(data_loader: DataLoader, models: List[torch.nn.Module], device: torch.device,
             subset_name: str, criterion, logistic_threshold: float,
             exp_dir: str, metric='accuracy', max_seq: int = -1, aggregate: str = 'add', aggregate_or_th: float = 0.5):
    assert aggregate in ['add', 'or']
    assert aggregate_or_th > 0 and aggregate_or_th <= 1
    assert metric in ['accuracy', 'f1']
    metric_name = metric
    if metric == 'accuracy':
        metric_score = metrics.accuracy_score
        metric_args = {}
        metric_other_name, metric_other_args, metric_other_score = 'f1', {'average': 'macro'}, metrics.f1_score
    else:
        metric_score = metrics.f1_score
        metric_args = {'average': 'macro'}
        metric_other_name, metric_other_args, metric_other_score = 'accuracy', {}, metrics.accuracy_score

    #  deterministic(seed)
    # seed should NOT be used here (TODO review)
    total = 0
    loss_total = 0
    [model.eval() for model in models]
    with torch.no_grad():
        predictions = [defaultdict(list) for _ in range(len(models))]
        patient_ids_from_start = defaultdict(list)

        corrects = defaultdict(list)
        first = True
        for model_idx, model in enumerate(models):
            for data in data_loader:
                patient_ids, static_data, dynamic_data, lengths, labels = data[0], data[1].to(device), \
                                                                          data[2], data[3].to(device), \
                                                                          data[4].to(device)

                if max_seq != -1:
                    new_dynamic_data = []
                    for data in dynamic_data:
                        new_dynamic_data.append(data[len(data) - max_seq:] if len(data) > max_seq else data)
                    dynamic_data = new_dynamic_data

                # TO FIX: the padding remove one sequence from the list!
                dynamic_data_padded = pad_sequence(dynamic_data, batch_first=True, padding_value=0).to(device)

                effective_lengths = torch.ones(dynamic_data_padded.shape[0]).long().to(device)
                c_lengths = torch.tensor(list(range(dynamic_data_padded.shape[1]))).long().to(device)
                outputs = torch.zeros(dynamic_data_padded.shape[0]).to(device)
                hidden = model.init_hidden(dynamic_data_padded.shape[0])
                max_seq_step = dynamic_data_padded.shape[1]
                dynamic_data_history = torch.zeros(len(data[0]), dynamic_data_padded.shape[1],
                                                   model.hidden_size).to(device)

                for seq_step in range(max_seq_step):
                    events = dynamic_data_padded[:, seq_step, :]
                    non_zero = (effective_lengths != 0).nonzero().squeeze()
                    static = static_data[non_zero]
                    lens = effective_lengths[non_zero]
                    evs = events[non_zero]
                    if len(lens.shape) != 1:
                        static = static.unsqueeze(dim=0)
                        lens = lens.unsqueeze(dim=0)
                        evs = evs.unsqueeze(dim=0)
                    evs = evs.unsqueeze(dim=1)
                    if model.arch != 'lstm':
                        if len(non_zero.shape) == 0:
                            outputs[non_zero], hidden[:, non_zero:non_zero + 1, :], dynamic_data_event, _, _ = model(
                                (static, evs,
                                 lens, hidden, dynamic_data_history), seq_step)
                        else:
                            outputs[non_zero], hidden[:, non_zero, :], dynamic_data_event, _, _ = model(
                                (static, evs, lens,
                                 hidden, dynamic_data_history), seq_step)
                    else:
                        outputs[non_zero], h, dynamic_data_event, _, _ = model(
                            (static, evs, lens, hidden, dynamic_data_history), seq_step)
                        if len(non_zero.shape) == 0:
                            hidden[0][:, non_zero:non_zero + 1, :] = h[0]
                            hidden[1][:, non_zero:non_zero + 1, :] = h[1]
                        else:
                            hidden[0][:, non_zero, :] = h[0]
                            hidden[1][:, non_zero, :] = h[1]

                    # append predictions
                    non_zero_indexes = non_zero.tolist() if isinstance(non_zero.tolist(), list) else [non_zero.tolist()]
                    # append predictions and patient ids from start (left-aligned sequences)
                    for pred_idx in non_zero_indexes:
                        pred = torch.sigmoid(outputs[pred_idx]).clone().data
                        pred_seq_len = lengths.tolist()[pred_idx] - 1
                        predictions[model_idx][seq_step].append(pred)

                        # furthermore, store the patient_ids for each step
                        pid = patient_ids[pred_idx]
                        patient_ids_from_start[seq_step].append(int(pid))

                    dynamic_data_history[:, seq_step, :] = dynamic_data_event
                    if first:
                        outs = labels[non_zero].clone().data.tolist()
                        outs = outs if isinstance(outs, list) else [outs]
                        for label in outs:
                            corrects[seq_step].append(label)
                        total += 1 if len(non_zero.shape) == 0 else len(non_zero)
                    if outputs[non_zero].size():
                        if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                            loss_total += criterion(outputs[non_zero].clone(), labels[non_zero].float())
                        else:
                            loss_total += criterion(torch.sigmoid(outputs[non_zero]).clone(), labels[non_zero].float())
                    effective_lengths = (c_lengths[seq_step] < lengths - 1).long()
            first = False
    loss_total /= len(models)

    # compute predictions and from end (right-aligned sequences) using the sequence length for each prediction
    max_steps = len(predictions[0].keys())

    # Compute voted predictions
    def aggregate_or(votes):
        return (1 if len(list(filter(lambda x: x == 1, votes)))/len(votes) >= aggregate_or_th else 0,
                sum(votes)/len(votes))

    predicted = defaultdict(list)
    predicted_probs = defaultdict(list)
    for step in range(max_steps):
        # for each step, sum the prediction of each model in the ensemble
        preds_votes = []
        if aggregate == 'add':
            for model_idx in range(len(predictions)):
                if len(preds_votes) == 0:
                    preds_votes = [pred.tolist() for pred in predictions[model_idx][step]]
                else:
                    preds_votes = list(map(add, preds_votes, [pred.tolist() for pred in predictions[model_idx][step]]))
            predicted[step] = [1 if pred >= logistic_threshold * len(models) else 0 for pred in preds_votes]
            predicted_probs[step] = preds_votes
        else:
            preds_votes_to_aggregate = []
            for model_idx in range(len(predictions)):
                if len(preds_votes_to_aggregate) == 0:
                    preds_votes_to_aggregate = [pred.tolist() for pred in predictions[model_idx][step]]
                    preds_votes_to_aggregate = [[1 if pred >= logistic_threshold else 0 for pred in
                                                preds_votes_to_aggregate]]
                else:
                    new_votes = [pred.tolist() for pred in predictions[model_idx][step]]
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
        lista_ids = patient_ids_from_start[step]
        lista_labels = corrects[step]
        lista_predicciones = predicted[step]
        lista_probs = predicted_probs[step]

        for id, label, prediction, prob in zip(lista_ids, lista_labels, lista_predicciones, lista_probs):
            if step == 0:
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
        predicted_all_scores.extend(predicted_probs[step])

    predicted_all = []
    for step in range(max_steps):
        predicted_all.extend(predicted[step])

    predicted_all_from_end = []
    for step in range(max_steps):
        predicted_all_from_end.extend(predicted_from_end[step])

    corrects_all = []
    for step in range(max_steps):
        corrects_all.extend(corrects[step])

    corrects_all_from_end = []
    for step in range(max_steps):
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

    return float(loss_total), output_table
