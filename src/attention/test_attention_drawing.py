import argparse
from utils import load_architecture, evaluate, ArgsStruct, deterministic
import torch
import os
import json
import random
from attention.explain_prediction import explain_prediction, show_attention_matrix, show_relevant_graph_features
from torch.nn.utils.rnn import pad_sequence
from training.covid_torch_dataset import CovidDataset, CovidData, DatasetBuilder
from torch.utils.data.dataloader import DataLoader
from data_analysis.extract_features import extract_features_information_metrics
import logging
import shutil
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import Counter


def attention_weights(args_eval):
    device = torch.device('cuda:0' if not args_eval.no_cuda and torch.cuda.is_available() else 'cpu')
    features = []
    if args_eval.features_file:
        if args_eval.features_top_n:
            features = extract_features_information_metrics(args_eval.features_file, args_eval.features_top_n)
        else:
            features = extract_features_information_metrics(args_eval.features_file)
    dataset = CovidData(args_eval.data_path, 'dataset_static.csv', 'dataset_dynamic.csv', 'dataset_labels.csv',
                        filter_patients_ids=[],
                        features_selected=features)
    '''
    with open('../baselines/train_patient_ids.json') as tpi_f:
        js =  json.load(tpi_f)
        tpi = [int(tpii) for tpii in js["test"]]
    '''

    dataset = CovidDataset(data=dataset)
    static_data_columns = list(pd.read_csv(os.path.join(args_eval.data_path, 'dataset_static.csv')).columns)
    data_loader = DataLoader(dataset, batch_size=args_eval.batch_size, shuffle=False, collate_fn=CovidDataset.collate)

    models = []
    models_path = []
    for model_path in args_eval.models_path:
        # TODO Make better fold iteration listing/ordering folds etc.
        for i in range(5):
            with open(os.path.join(model_path, 'args.json'), 'r') as f:
                train_args = ArgsStruct(**json.load(f))
            args = train_args
            model = load_architecture(device, args)
            if torch.cuda.is_available() and not args.no_cuda:
                map_location = lambda storage, loc: storage.cuda()
            else:
                map_location = lambda storage, location: 'cpu'

            model.load_state_dict(torch.load(os.path.join(model_path, f'checkpoint_best_{i}.pt')))
            model.to(device)
            models.append(model)
            models_path.append(model_path)

    deterministic(args_eval.eval_seed)
    [model.eval() for model in models]
    with torch.no_grad():
        for idx, model in enumerate(models):
            model_path = models_path[idx]
            peaks = list()
            peaks_no = list()
            dynamic_embeddings = list()
            static_dynamic_attention_weights_sum = None
            dynamic_dynamic_attention_weights_sum = None
            day_count = None
            for data in data_loader:
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
                dynamic_dynamic_w_history = torch.zeros(len(data[0]), max_seq_len, max_seq_len)

                static_dynamic_attention_weights_sum_batch = torch.zeros(1, static_data.shape[1], model.hidden_size)
                dynamic_dynamic_attention_weights_sum_batch = torch.zeros(1, max_seq_len, max_seq_len)
                day_count_batch = torch.zeros(1, max_seq_len, 1)
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
                            outputs[non_zero], hidden[:, non_zero:non_zero + 1,
                                               :], dynamic_data_event, attention_w_static_dynamic, attention_w_dynamic_dynamic = model(
                                (static, evs, lens, hidden, dynamic_data_history), seq_step)
                        else:
                            outputs[non_zero], hidden[:, non_zero,
                                               :], dynamic_data_event, attention_w_static_dynamic, attention_w_dynamic_dynamic = \
                                model((static, evs, lens, hidden, dynamic_data_history), seq_step)

                    else:
                        outputs[
                            non_zero], h, dynamic_data_event, attention_w_static_dynamic, attention_w_dynamic_dynamic = \
                            model((static, evs, lens, hidden, dynamic_data_history), seq_step)

                        if len(non_zero.shape) == 0:
                            hidden[0][:, non_zero:non_zero + 1, :] = h[0]
                            hidden[1][:, non_zero:non_zero + 1, :] = h[1]
                        else:
                            hidden[0][:, non_zero, :] = h[0]
                            hidden[1][:, non_zero, :] = h[1]

                    dynamic_data_history[:, seq_step, :] = dynamic_data_event
                    dynamic_data_history.to(device)
                    effective_lengths = (c_lengths[seq_step] <= lengths).long()
                    dynamic_dynamic_w_history[:, seq_step, :] = attention_w_dynamic_dynamic[:, 0, :]

                    if model.use_attention and labels[non_zero].shape:
                        day_count_batch[0, seq_step, 0] = len(non_zero)
                        if model.attention_fields == 'both':
                            static_dynamic_attention_weights_sum_batch = torch.unsqueeze(torch.sum(torch.cat(
                                (attention_w_static_dynamic[
                                     [l_idx for l_idx, l in enumerate(labels[non_zero]) if l == 1]],
                                 static_dynamic_attention_weights_sum_batch), dim=0), dim=0), dim=0)
                            if seq_step > 0:
                                dynamic_dynamic_attention_weights_sum_batch[0][seq_step][:seq_step - 1] = torch.squeeze(
                                    torch.sum(attention_w_dynamic_dynamic[
                                                  [l_idx for l_idx, l in enumerate(labels[non_zero]) if l == 1]],
                                              dim=0), dim=0)[:seq_step - 1] + \
                                                                                                          dynamic_dynamic_attention_weights_sum_batch[
                                                                                                              0][
                                                                                                              seq_step][
                                                                                                          :seq_step - 1]
                        elif model.attention_fields == 'static_dynamic':
                            static_dynamic_attention_weights_sum_batch = torch.unsqueeze(torch.sum(torch.cat(
                                (attention_w_static_dynamic[
                                     [l_idx for l_idx, l in enumerate(labels[non_zero]) if l == 1]],
                                 static_dynamic_attention_weights_sum_batch), dim=0), dim=0), dim=0)
                        else:
                            if seq_step > 0:
                                dynamic_dynamic_attention_weights_sum_batch[0][seq_step][:seq_step] = torch.squeeze(
                                    torch.sum(attention_w_dynamic_dynamic[
                                                  [l_idx for l_idx, l in enumerate(labels[non_zero]) if l == 1]],
                                              dim=0), dim=0)[:seq_step] + \
                                                                                                          dynamic_dynamic_attention_weights_sum_batch[
                                                                                                              0][
                                                                                                              seq_step][
                                                                                                          :seq_step]

                if model.use_attention:
                    if day_count is not None:
                        max_shape_1 = max(day_count.shape[1], day_count_batch.shape[1])
                        mask = torch.zeros(1, max_shape_1, 1)
                        mask[:, :day_count.shape[1], :] = day_count
                        day_count = mask
                        mask = torch.zeros(1, max_shape_1, 1)
                        mask[:, :day_count_batch.shape[1], :] = day_count_batch
                        day_count_batch = mask
                        day_count = torch.unsqueeze(
                            torch.sum(
                                torch.cat(
                                    (day_count, day_count_batch),
                                    dim=0), dim=0), dim=0)
                    else:
                        day_count = day_count_batch
                    if static_dynamic_attention_weights_sum is not None:
                        static_dynamic_attention_weights_sum = torch.unsqueeze(
                            torch.sum(
                                torch.cat(
                                    (static_dynamic_attention_weights_sum, static_dynamic_attention_weights_sum_batch),
                                    dim=0), dim=0), dim=0)
                    else:
                        static_dynamic_attention_weights_sum = static_dynamic_attention_weights_sum_batch
                    if dynamic_dynamic_attention_weights_sum is not None:
                        max_shape_1 = max(dynamic_dynamic_attention_weights_sum.shape[1],
                                          dynamic_dynamic_attention_weights_sum_batch.shape[1])
                        max_shape_2 = max(dynamic_dynamic_attention_weights_sum.shape[2],
                                          dynamic_dynamic_attention_weights_sum_batch.shape[2])
                        mask = torch.zeros(1, max_shape_1, max_shape_2)
                        mask[:, :dynamic_dynamic_attention_weights_sum.shape[1],
                        :dynamic_dynamic_attention_weights_sum.shape[2]] = dynamic_dynamic_attention_weights_sum
                        dynamic_dynamic_attention_weights_sum = mask
                        mask = torch.zeros(1, max_shape_1, max_shape_2)
                        mask[:, :dynamic_dynamic_attention_weights_sum_batch.shape[1],
                        :dynamic_dynamic_attention_weights_sum_batch.shape[
                            2]] = dynamic_dynamic_attention_weights_sum_batch
                        dynamic_dynamic_attention_weights_sum_batch = mask
                        dynamic_dynamic_attention_weights_sum = torch.unsqueeze(
                            torch.sum(
                                torch.cat(
                                    (
                                    dynamic_dynamic_attention_weights_sum, dynamic_dynamic_attention_weights_sum_batch),
                                    dim=0), dim=0), dim=0)
                    else:
                        dynamic_dynamic_attention_weights_sum = dynamic_dynamic_attention_weights_sum_batch

                    if dynamic_dynamic_w_history is not None:
                        for b in range(dynamic_dynamic_w_history.shape[0]):
                            peak_days = []
                            length = lengths[b]
                            rnn_days = dynamic_data_history[b].numpy()
                            for d in range(dynamic_dynamic_w_history.shape[1]):
                                pds = list(find_peaks(dynamic_dynamic_w_history[b][d].numpy())[0])
                                peak_days.extend(pds)
                            if peak_days:
                                peaks.extend([[patient_ids[b], peak, int(labels[b]), False] for peak in set(peak_days) if peak < length])

                                count_peak_days = Counter(peak_days)
                                count_peak_days = {k:v/(float(length)-k) for k, v in count_peak_days.items() if k < length}
                                peaks_filtered = [[patient_ids[b], peak, int(labels[b]), True] for peak in
                                                     [list(count_peak_days.keys())[i] for i in
                                                      find_peaks(list(count_peak_days.values()))[0]]]

                                if not len(peaks_filtered):
                                    peaks_no.append(patient_ids[b])
                                else:
                                    peaks.extend(peaks_filtered)

                                peak_days = [pf[1] for pf in peaks_filtered]
                                for i in range(length):
                                    rnn_day = list(rnn_days[i])
                                    rnn_day.insert(0, 1 if i in peak_days else 0)
                                    rnn_day.insert(0, patient_ids[b])
                                    dynamic_embeddings.append(rnn_day)
                            else:
                                peaks_no.append(patient_ids[b])
                                for rnn_day in rnn_days:
                                    rnn_day = list(rnn_day)
                                    rnn_day.insert(0, 0)
                                    rnn_day.insert(0, patient_ids[b])
                                    dynamic_embeddings.extend(rnn_day)

            if model.use_attention:
                if model.attention_fields == 'static_dynamic':
                    static_dynamic_attention_weights_sum = torch.squeeze(static_dynamic_attention_weights_sum, dim=0)
                    relevant_features = explain_prediction(None,
                                                           None,
                                                           static_dynamic_attention_weights_sum,
                                                           output_path=model_path,
                                                           fname='all_static',
                                                           input1_labels=static_data_columns,
                                                           big=True)
                    print("Most relevant variables are:", relevant_features)
                    show_relevant_graph_features(relevant_features, output_path=model_path, fname='relevant_features')
                    show_attention_matrix(static_dynamic_attention_weights_sum, static_data_columns,
                                          output_path=model_path, fname='all_static_matrix')
                elif model.attention_fields == 'dynamic_dynamic':
                    dynamic_dynamic_attention_weights_sum = torch.div(
                        torch.squeeze(dynamic_dynamic_attention_weights_sum, dim=0), torch.squeeze(day_count, dim=0))
                    explain_prediction(None,
                                       None,
                                       dynamic_dynamic_attention_weights_sum,
                                       output_path=model_path, draw_all_labels=True,
                                       fname='all_dynamic',
                                       input1_labels=['day ' + str(i)
                                                      for i in
                                                      range(1, dynamic_dynamic_attention_weights_sum.shape[0] + 1)],
                                       input2_labels=['day ' + str(i)
                                                      for i in
                                                      range(1, dynamic_dynamic_attention_weights_sum.shape[0] + 2)],
                                       limit=70)
                    pd.DataFrame(peaks, columns=['patientid', 'peak_day', 'label', 'significant']).to_csv(os.path.join(model_path, f'peaks_{idx}.csv'))
                    pd.DataFrame(peaks_no, columns=['patientid']).to_csv(os.path.join(model_path, f'no_peaks_{idx}.csv'))
                    pd.DataFrame(dynamic_embeddings).to_csv(os.path.join(model_path, f'dynamic_embeddings_{idx}.csv'))
                else:
                    static_dynamic_attention_weights_sum = torch.squeeze(static_dynamic_attention_weights_sum, dim=0)
                    relevant_features = explain_prediction(None, None,
                                                           torch.squeeze(static_dynamic_attention_weights_sum, dim=0),
                                                           output_path=model_path, fname='all_static',
                                                           input1_labels=static_data_columns,
                                                           big=True)
                    show_relevant_graph_features(relevant_features, output_path=model_path, fname='relevant_features')
                    print("Most relevant variables are:", relevant_features)
                    show_attention_matrix(static_dynamic_attention_weights_sum, static_data_columns,
                                          output_path=model_path, fname='all_static_matrix')
                    dynamic_dynamic_attention_weights_sum = torch.div(
                        torch.squeeze(dynamic_dynamic_attention_weights_sum, dim=0), torch.squeeze(day_count, dim=0))
                    explain_prediction(None, None, dynamic_dynamic_attention_weights_sum,
                                       output_path=model_path, draw_all_labels=True, fname='all_dynamic',
                                       input1_labels=['day ' + str(i) for i in
                                                      range(1, dynamic_dynamic_attention_weights_sum.shape[0] + 1)],
                                       input2_labels=['day ' + str(i) for i in
                                                      range(1, dynamic_dynamic_attention_weights_sum.shape[0] + 2)],
                                       limit=70)


def test_attention(args_eval):
    device = torch.device('cuda:0' if not args_eval.no_cuda and torch.cuda.is_available() else 'cpu')
    features = []
    if args_eval.features_file:
        if args_eval.features_top_n:
            features = extract_features_information_metrics(args_eval.features_file, args_eval.features_top_n)
        else:
            features = extract_features_information_metrics(args_eval.features_file)
    dataset = CovidData(args_eval.data_path, 'dataset_static.csv', 'dataset_dynamic.csv', 'dataset_labels.csv',
                        filter_patients_ids=[],
                        features_selected=features)
    dataset = CovidDataset(data=dataset, subset='test')
    static_data_columns = list(pd.read_csv(os.path.join(args_eval.data_path, 'dataset_static.csv')).columns)
    static_data_columns = [std for std in static_data_columns if std in features]
    data_loader = DataLoader(dataset, batch_size=args_eval.batch_size, shuffle=False, collate_fn=CovidDataset.collate)

    models = []
    for model_path in args_eval.models_path:
        with open(os.path.join(model_path, 'args.json'), 'r') as f:
            train_args = ArgsStruct(**json.load(f))
        args = train_args
        model = load_architecture(device, args)
        if torch.cuda.is_available() and not args.no_cuda:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = lambda storage, location: 'cpu'
        model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint_best.pt')))
        model.to(device)
        models.append(model)

    deterministic(args_eval.eval_seed)
    total = 0
    loss_total = 0
    [model.eval() for model in models]
    with torch.no_grad():
        for idx, model in enumerate(models):
            for data in data_loader:
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
                            outputs[non_zero], hidden[:, non_zero:non_zero + 1,
                                               :], dynamic_data_event, attention_w_static_dynamic, attention_w_dynamic_dynamic = model(
                                (static, evs, lens, hidden, dynamic_data_history), seq_step)
                        else:
                            outputs[non_zero], hidden[:, non_zero,
                                               :], dynamic_data_event, attention_w_static_dynamic, attention_w_dynamic_dynamic = \
                                model((static, evs, lens, hidden, dynamic_data_history), seq_step)

                    else:
                        outputs[
                            non_zero], h, dynamic_data_event, attention_w_static_dynamic, attention_w_dynamic_dynamic = \
                            model((static, evs, lens, hidden, dynamic_data_history), seq_step)

                        if len(non_zero.shape) == 0:
                            hidden[0][:, non_zero:non_zero + 1, :] = h[0]
                            hidden[1][:, non_zero:non_zero + 1, :] = h[1]
                        else:
                            hidden[0][:, non_zero, :] = h[0]
                            hidden[1][:, non_zero, :] = h[1]

                    dynamic_data_history[:, seq_step, :] = dynamic_data_event
                    dynamic_data_history.to(device)
                    effective_lengths = (c_lengths[seq_step] <= lengths).long()

                    print('predictions: ', outputs[0])
                    if seq_step <= 1:
                        continue
                    FIRST_LOOP_PATIENT_IDX = 0
                    if model.use_attention:
                        if model.attention_fields == 'static_dynamic':
                            explain_prediction(static[FIRST_LOOP_PATIENT_IDX],
                                               evs.unsqueeze(dim=1)[FIRST_LOOP_PATIENT_IDX],
                                               attention_w_static_dynamic[FIRST_LOOP_PATIENT_IDX],
                                               output_path=model_path,
                                               fname='static' + str(seq_step),
                                               input1_labels=static_data_columns,
                                               big=True)
                        elif model.attention_fields == 'dynamic_dynamic':
                            explain_prediction(evs.unsqueeze(dim=1)[FIRST_LOOP_PATIENT_IDX],
                                               evs.unsqueeze(dim=1)[FIRST_LOOP_PATIENT_IDX],
                                               torch.unsqueeze(attention_w_dynamic_dynamic[
                                                                   FIRST_LOOP_PATIENT_IDX][0][0:seq_step], dim=0),
                                               output_path=model_path, draw_all_labels=True,
                                               fname='dynamic' + str(seq_step),
                                               input1_labels=['day ' + str(seq_step + 1)],
                                               input2_labels=['day ' + str(i) for i in range(1, seq_step + 1)])
                        else:
                            explain_prediction(static[FIRST_LOOP_PATIENT_IDX],
                                               evs.unsqueeze(dim=1)[FIRST_LOOP_PATIENT_IDX],
                                               attention_w_static_dynamic[FIRST_LOOP_PATIENT_IDX],
                                               output_path=model_path,
                                               fname='static' + str(seq_step),
                                               input1_labels=static_data_columns,
                                               big=True)
                            explain_prediction(evs.unsqueeze(dim=1)[FIRST_LOOP_PATIENT_IDX],
                                               evs.unsqueeze(dim=1)[FIRST_LOOP_PATIENT_IDX],
                                               torch.unsqueeze(attention_w_dynamic_dynamic[
                                                                   FIRST_LOOP_PATIENT_IDX][0][0:seq_step], dim=0),
                                               model_path, draw_all_labels=True,
                                               fname='dynamic' + str(seq_step),
                                               input1_labels=['day ' + str(seq_step + 1)],
                                               input2_labels=['day ' + str(i) for i in range(1, seq_step + 1)])
                        print('attention weights drawn')

                break
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('models_path', type=str, help='Path to model directory. If more than one path is provided, an'
                                                      'ensemble of models is loaded', nargs='+')
    parser.add_argument('data_path', type=str, help='Path to data')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=32)
    parser.add_argument('--subset', type=str, help='Subset to evaluate on (train, valid, test)', default='test')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--data-seed', type=int, help='Random seed for data', default=42)
    parser.add_argument('--eval-seed', type=int, help='Random seed for evaluation', default=42)
    parser.add_argument('--logistic-threshold', type=float, help='Threshold of the logistic regression (default 0.5)',
                        default=0.5)
    parser.add_argument('--no-stratified', action='store_true', help='Disables stratified split')
    parser.add_argument('--all', action='store_true', help='Get whole attention weight graphs or test '
                                                           'attention for a user', default=False)
    parser.add_argument('--features-file', type=str,
                        help='File with a list of features ranked with information metrics')
    parser.add_argument('--features-top-n', type=int, help='Top N common features to select')

    args_eval = parser.parse_args()

    if args_eval.all:
        attention_weights(args_eval)
    else:
        test_attention(args_eval)
