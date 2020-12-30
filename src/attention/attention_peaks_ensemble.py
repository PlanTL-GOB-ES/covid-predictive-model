import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

PATH_TO_CSV = ""
PATH_TO_PATIENT_SELECTION = ""
DAY_OFFSET = 0
N_FEATURES = []
DEATH = False
directories = []

if __name__ == '__main__':
    with open(PATH_TO_PATIENT_SELECTION) as f:
        patients = json.load(f)['test']

    ratios_peaks = []
    ratios_patients_peaks = []
    peak_matches = defaultdict(lambda: defaultdict(int))
    for i in range(5):
        for directory in directories:
            df_peaks = pd.read_csv(os.path.join(PATH_TO_CSV, directory, f'peaks_{i}.csv'))
            df_peaks_no = pd.read_csv(os.path.join(PATH_TO_CSV, directory, f'no_peaks_{i}.csv'))
            df_dynamic_embedding = pd.read_csv(os.path.join(PATH_TO_CSV, directory, f'dynamic_embeddings_{i}.csv'))
            df_peaks = df_peaks[df_peaks['patientid'].isin(patients)]
            df_peaks_no = df_peaks_no[df_peaks_no['patientid'].isin(patients)]

            # Filter
            n_peaks = len(df_peaks)
            df_peaks = df_peaks[df_peaks['significant']]
            ratios_peaks.append(len(df_peaks) / n_peaks)
            ratios_patients_peaks.append(len(df_peaks['patientid'].unique()) / len(patients))

            # Set matching
            for _, peak in df_peaks.iterrows():
                peak_matches[peak['patientid']][peak['day']] = peak_matches[peak['patientid']][peak['day']] + 1

    print('Statistics')
    print(np.mean(ratios_peaks), np.std(ratios_peaks))
    print(np.mean(ratios_patients_peaks), np.std(ratios_patients_peaks))

    divider = len(directories) * 5
    print("Peak match ratio:",
          sum([max(peak_matches[pm_key].values()) / divider for pm_key in peak_matches.keys()]) / len(patients))
