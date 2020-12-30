from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os
from math import floor, ceil

color_grey = '#B3B3B3'
color_blue = 'cornflowerblue'
color_red = '#D62728'
color_green = '#60A917'

# ================    ===============================
# character           description
# ================    ===============================
#    -                solid line style
#    --               dashed line style
#    -.               dash-dot line style
#    :                dotted line style
#    .                point marker
#    ,                pixel marker
#    o                circle marker
#    v                triangle_down marker
#    ^                triangle_up marker
#    <                triangle_left marker
#    >                triangle_right marker
#    1                tri_down marker
#    2                tri_up marker
#    3                tri_left marker
#    4                tri_right marker
#    s                square marker
#    p                pentagon marker
#    *                star marker
#    h                hexagon1 marker
#    H                hexagon2 marker
#    +                plus marker
#    x                x marker
#    D                diamond marker
#    d                thin_diamond marker
#    |                vline marker
#    _                hline marker
# ================    ===============================


def plot_accuracy(x_data, rf_data, rnn_data, examples):
    plt.style.use('default')

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Days')
    ax1.set_ylabel('Accuracy')

    ax1.tick_params(axis='y')
    ax1.set_xticks(np.arange(min(x_data), max(x_data) + 1, 1))
    ax1.set_yticks(np.arange(floor(min(rf_data + rnn_data)), 101.0, 1))

    plt.plot(x_data, rf_data, color_green, marker='^')
    rf_acc = mpatches.Patch(color=color_green, label='Random Forest')

    plt.plot(x_data, rnn_data, color_red, marker='o')
    rnn_acc = mpatches.Patch(color=color_red, label='RNN')
    plt.legend(handles=[rf_acc, rnn_acc])

    plt.savefig(os.path.join(script_dir, 'accuracy_plot.png'))


def plot_determinations(determinations):

    ind = np.arange(len(variable))

    fig, ax1 = plt.subplots()

    width = 0.80
    determinations.plot(kind='bar', y='Frequencies', width=width, color=color_green, ax=ax1)
    ax1.set_xlabel('Laboratory determinations')
    ax1.set_xticks(np.arange(0, 401, 50))
    ax1.set_xticklabels(np.arange(0, 401, 50))
    ax1.set_yticks(np.arange(0, 18001, 1000))
    ax1.set_ylabel('Frequency')
    ax1.tick_params(axis='y')

    freq_leg = mpatches.Patch(color=color_green, label='Determination frequencies')
    cum_leg = mpatches.Patch(color=color_red, label='Cumulative line')
    plt.legend(handles=[freq_leg, cum_leg], bbox_to_anchor=(0.49, 0.9))

    ax2 = ax1.twinx()

    ax2.set_ylabel('Cumulative frequency')
    ax2.plot(ind, determinations[['Cumulative line']], color=color_red)
    ax2.tick_params(axis='y')

    plt.subplots_adjust(bottom=0.15)
    fig.tight_layout()

    plt.savefig(os.path.join(script_dir, 'laboratory_determinations.png'))


def plot_determinations_by_bins(determinations_by_bins):

    fig, ax1 = plt.subplots()

    width = 0.80
    ax1 = determinations_by_bins.plot(kind='bar', width=width, color=color_green, figsize=(12, 6))
    ax1.set_xlabel('Bins')
    ax1.set_xticklabels(determinations_by_bins.index, rotation=45)
    plt.setp(ax1.xaxis.get_majorticklabels(), ha='right')
    ax1.set_yticks(np.arange(0, 351, 50))
    ax1.set_ylabel('Frequency')
    ax1.tick_params(axis='y')

    freq_leg = mpatches.Patch(color=color_green, label='Determination frequencies by bins')
    plt.legend(handles=[freq_leg])

    plt.subplots_adjust(bottom=0.3)
    fig.tight_layout()

    plt.savefig(os.path.join(script_dir, 'laboratory_determinations_bybins.png'))


def plot_demographic(demographic):

    fig, ax1 = plt.subplots()

    demographic.plot(kind='line', x='Bin', y='Female', color=color_green, ax=ax1)
    demographic.plot(kind='line', x='Bin', linestyle='dashed', y='Male', color=color_green, ax=ax1)
    demographic.plot(kind='line', x='Bin', y='ICU female', color='orange', ax=ax1)
    demographic.plot(kind='line', x='Bin', linestyle='dashed', y='ICU male', color='orange', ax=ax1)
    demographic.plot(kind='line', x='Bin', y='Death female', color=color_red, ax=ax1)
    demographic.plot(kind='line', x='Bin', linestyle='dashed', y='Death male', color=color_red, ax=ax1)

    ax1.set_xlabel('Age bins')
    ax1.set_yticks(np.arange(0, 241, 20))
    ax1.set_ylabel('Counts')
    ax1.tick_params(axis='y')

    fig.tight_layout()

    plt.savefig(os.path.join(script_dir, 'demographic.png'))


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.realpath(__file__))
    '''
    ###############################################################
    df = pd.read_csv('scores/accuracies_per_day.tsv', sep='\t')

    days = [int(d) for d in df['day']]
    acc_by_days_rf = [float(a.strip("%")) for a in df['acc_rf']]
    acc_by_days_rnn = [float(a.strip("%")) for a in df['acc_rnn']]
    examples_by_days = [int(e) for e in df['examples']]

    plot_accuracy(days, acc_by_days_rf, acc_by_days_rnn, examples_by_days)
    '''
    ###############################################################
    laboratory = pd.read_csv('scores/obx_lab_metrics.csv', sep=',')
    variable = [v for v in laboratory['OBX']]
    freq = [int(f) for f in laboratory['counts']]
    cumulative_freq = (np.cumsum(freq)/np.sum(freq))*100

    determinations = pd.DataFrame({'Variables': variable,
                                   'Frequencies': freq,
                                   'Cumulative line': cumulative_freq
                                   })
    plot_determinations(determinations)

    ###############################################################
    determinations_by_bins = pd.cut(freq, range(0, 18001, 500)).value_counts(dropna=False).sort_index().to_frame()
    plot_determinations_by_bins(determinations_by_bins)

    ###############################################################
    '''
    demo = pd.read_csv('scores/demographic_features.tsv', sep='\t')

    bin = [b for b in demo['Bin']]
    female = [f for f in demo['Female']]
    male = [m for m in demo['Male']]
    icu_female = [f for f in demo['ICU female']]
    icu_male = [m for m in demo['ICU male']]
    death_female = [f for f in demo['Death female']]
    death_male = [m for m in demo['Death male']]

    demographic = pd.DataFrame({'Bin': bin,
                                   'Female': female,
                                   'Male': male,
                                   'ICU female': icu_female,
                                   'ICU male': icu_male,
                                   'Death female': death_female,
                                   'Death male': death_male
                                })
    plot_demographic(demographic)
    '''
