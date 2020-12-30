import os
import matplotlib.pyplot as plt
import numpy as np

data = {
    'accuracy': {'HM_RNN': 0.8896, 'HM_RF': 0.9032, 'HM_SVC': 0.8889, 'H12O_RNN': 0.9011, 'H12O_RF': 0.9341, 'H12O_SVC': 0.9203},
    'accuracy_dev': {'HM_RNN': 0.0054, 'HM_RF': 0.0032, 'HM_SVC': 0.0035, 'H12O_RNN': 0.0106, 'H12O_RF': 0.0010, 'H12O_SVC': 0.0005},
    'sensitivity': {'HM_RNN': 0.8450, 'HM_RF': 0.6169, 'HM_SVC': 0.7806, 'H12O_RNN': 0.8082, 'H12O_RF': 0.6255, 'H12O_SVC': 0.6397},
    'sensitivity_dev': {'HM_RNN': 0.0075, 'HM_RF': 0.0109, 'HM_SVC': 0.0121, 'H12O_RNN': 0.0163, 'H12O_RF': 0.0102, 'H12O_SVC': 0.0087},
    'specificity': {'HM_RNN': 0.8998, 'HM_RF': 0.9687, 'HM_SVC': 0.9137, 'H12O_RNN': 0.9152, 'H12O_RF': 0.9806, 'H12O_SVC': 0.9626},
    'specificity_dev': {'HM_RNN': 0.0073, 'HM_RF': 0.0018, 'HM_SVC': 0.0028, 'H12O_RNN': 0.0163, 'H12O_RF': 0.0015, 'H12O_SVC': 0.0011},
    'f1': {'HM_RNN': 0.8351, 'HM_RF': 0.8228, 'HM_SVC': 0.8269, 'H12O_RNN': 0.8119, 'H12O_RF': 0.8380, 'H12O_SVC': 0.8161},
    'f1_dev': {'HM_RNN': 0.0063, 'HM_RF': 0.0061, 'HM_SVC': 0.0055, 'H12O_RNN': 0.0115, 'H12O_RF': 0.0030, 'H12O_SVC': 0.0021},
}

metric_keys = ['accuracy', 'sensitivity', 'specificity', 'f1']
metric_abbr = ['acc.', 'sen.', 'spe.', 'f1']
comparison_keys = ['HM', 'H12O']
model_keys = ['RNN', 'RF', 'SVC']

model_colors = {
    'RF': '#60A917',
    'SVC': 'cornflowerblue',
    'RNN': 'orange'
}

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))

    xtick_labels = list()
    xticks = list()
    fig = plt.figure(figsize=(6, 4))
    x_idx = 1
    for metric_idx, metric_key in enumerate(metric_keys):
        for model_key in model_keys:
            xs = [x_idx, x_idx + 1]

            ys = [data[metric_key][f'{comparison_key}_{model_key}'] for comparison_key in comparison_keys]
            plt.plot(xs, ys, color=model_colors[model_key], label=model_key if metric_idx == 0 else '', marker='|')

            ys_std = [data[f'{metric_key}_dev'][f'{comparison_key}_{model_key}'] for comparison_idx, comparison_key in
                      enumerate(comparison_keys)]
            plt.fill_between(xs, list(map(lambda x: x[0] + x[1], zip(ys, ys_std))),
                             list(map(lambda x: x[0] - x[1], zip(ys, ys_std))),
                             alpha=.1, color=model_colors[model_key])
        xtick_labels.extend(list(map(lambda x: f'{x} - {metric_abbr[metric_idx]}', comparison_keys)))
        xticks.extend([x_idx, x_idx + 1])
        x_idx = x_idx + 1 + 1
    plt.xticks(xticks, xtick_labels, rotation=45)
    plt.ylim(0.4, 1.0)
    plt.xlim(0.0, 9)
    plt.xlabel('Dataset - metric')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'comparison.png'))
