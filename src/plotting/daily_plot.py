from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

colors = ['#60A917', 'cornflowerblue', 'orange', '#D62728']
line_styles = ["-", "--"]

def plot_daily(data):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), )

    for i, metric in enumerate([[["Accuracy", "Accpm"]], [["Sensitivity", "Sensitivitypm"], ["Specificity", "Specificitypm"]],
                   [["F1", "F1pm"]]]):
        for j, plot in enumerate(['admission', 'outcome']):
            title = f'{" and ".join(list(map(lambda x: x[0], metric)))} day-by-day from {plot}'
            ax1_ = ax[i, j]
            for model_idx, model in enumerate(['rf', 'svm', 'rnn']):
                model_data = data[model][plot]
                for metric_idx, field in enumerate(metric):
                    x = list(model_data["Day"])
                    y = list(model_data[field[0]])
                    y_std = list(model_data[field[1]])

                    if len(metric) > 1:
                        label = model.upper() + "_" + field[0].lower()
                    else:
                        label = model.upper()
                    ax1_.plot(x, y, color=colors[model_idx], linestyle=line_styles[metric_idx], label=label)
                    ax1_.fill_between(x, [i - y_std for i, y_std in zip(y, y_std)],
                                    [i + y_std for i, y_std in zip(y, y_std)],
                                    alpha=.1, color=colors[model_idx])

            ax1_.set_xticks(np.arange(1, 21, 1))
            ax1_.set_xlabel('Day')
            ax1_.set_yticks(np.arange(0, 1.1, 0.1))
            ax1_.set_ylabel('Performance')
            ax1_.tick_params(axis='y')

            ax1_.set_title(title)
            ax1_.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, ncol=3)
            #ax1_.legend(loc='center left', bbox_to_anchor=(1, 0.5))


            #handles, labels = ax1_.get_legend_handles_labels()
            #ax1_.get_legend().remove()
            #fig.legend(handles, labels, loc='lower right', framealpha=1)

    fig.tight_layout()
    # plt.show()

    plt.savefig(os.path.join(script_dir, 'daily.png'))


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data = {}
    for model in ['rf', 'svm', 'rnn']:
        data[model] = {}
        for plot in ['admission', 'outcome']:
            print(model, plot)
            data[model][plot] = pd.read_csv(os.path.join(script_dir, 'scores', 'day_by_day',
                                                         model + '_' + plot + '.csv'),
                                            delim_whitespace=True)
    plot_daily(data)
