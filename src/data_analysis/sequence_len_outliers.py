import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json

import argparse


def statistics_lengths(file):
    df = pd.read_csv(file)
    df_days_from_inpat = df.groupby(['patientid']).size()
    seq_lens = [n for n in df_days_from_inpat]
    seq_lens_filtered = [n for n in df_days_from_inpat if n > 1]
    df_in_uci = df.groupby(['patientid'])['in_uci'].sum()
    in_uci = [n for n in df_in_uci if n > 0.0]

    # Plot sequence lengths outliers using the interquartile range
    # https://stackoverflow.com/questions/39068214/how-to-count-outliers-for-all-columns-in-python
    Q1 = np.percentile(seq_lens_filtered, 25)
    Q3 = np.percentile(seq_lens_filtered, 75)
    IQR = Q3 - Q1
    outliers = [n for n in seq_lens_filtered
                        if n < (Q1 - 1.5 * IQR) or n > (Q3 + 1.5 * IQR)]
    num_outliers = len(outliers)

    num_total = len(df_days_from_inpat)
    red_square = dict(markerfacecolor='r', marker='s')
    red_patch = mpatches.Patch(color='red', label=f'Outliers ({round((num_outliers / num_total) * 100, 2)} % )')
    fig5, ax5 = plt.subplots()
    ax5.set_title('Boxplot for sequence length ')
    plt.legend(handles=[red_patch])
    ax5.boxplot(seq_lens_filtered, vert=False, flierprops=red_square)
    plt.xlabel('days')
    plt.grid()
    plt.xticks(np.arange(min(seq_lens_filtered), max(seq_lens_filtered), 5))
    plt.xlim(0, 100)
    plt.savefig('boxplot_lengths.png')
    plt.show()

    # Save the outliers
    pids_to_len_outliers = {pid: l for pid, l in df_days_from_inpat.to_dict().items()
                            if l in outliers}
    with open('pid_to_len_outliers.json', 'w') as fn:
        json.dump(pids_to_len_outliers, fn)

    # Plot the sequence length histogram
    plt.hist(seq_lens)
    plt.grid()
    plt.xticks(np.arange(min(seq_lens), max(seq_lens), 5))
    plt.title('Histogram for sequence length')
    plt.xlim(0, 100)
    plt.yscale('log')
    plt.xlabel('days')
    plt.ylabel('frequency (log scale)')
    plt.savefig('histogram_lengths.png')
    plt.show()

    # Plot the days in uci histogram
    red_patch = mpatches.Patch(color='red', label=f'Patient in UCI ({round((len(in_uci) / num_total) * 100, 2)}% )')
    fig6, ax6 = plt.subplots()
    ax6.set_title('Histogram for days in uci')
    plt.legend(handles=[red_patch])
    plt.hist(in_uci)
    plt.grid()
    plt.xticks(np.arange(min(in_uci), max(in_uci), 2))
    plt.xlim(0, max(in_uci))
    plt.xlabel('days')
    plt.ylabel('frequency')
    plt.savefig('histogram_lengths_in_uci.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamic_features', type=str, help='File .csv with the dynamic features')
    args = parser.parse_args()
    file = '../../src/data_processing/data_features_label_death/dataset_dynamic.csv'
    statistics_lengths(file)
