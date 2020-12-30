import os
import glob
import json
from shutil import copyfile

CLASSIFIER = 'rf'
DATA_NAME = 'death_v2'
DATA_TYPE = ['base', 'with_imputation', 'with_imputation_with_reference_values', 'with_missing',
            'with_missing_with_imputation', 'with_missing_with_imputation_with_reference_values',
            'with_missing_with_reference_values', 'with_reference_values']
METRICS = ['f1', 'acc']

EXPERIMENTS_DIR = '/home/skamler/Projects/COVID-19/hm/covid-predictive-model/baselines/hm/'
OUT_DIR = f'/home/skamler/Projects/COVID-19/hm/covid-predictive-model/baselines/hm/experiment_{CLASSIFIER}'
BEST_DIR = '/home/skamler/Projects/COVID-19/hm/covid-predictive-model/baselines/hm/best.json'

os.makedirs(os.path.join(EXPERIMENTS_DIR, f'experiments_{CLASSIFIER}'), exist_ok=True)

best_models = []
for data in DATA_TYPE:
    if os.path.isdir(os.path.join(EXPERIMENTS_DIR, f'{DATA_NAME}_{data}')):
        for metric in METRICS:
            best = glob.glob(
                os.path.join(EXPERIMENTS_DIR, f'{DATA_NAME}_{data}', f'output_{CLASSIFIER}', f'test_{metric}_*_0_0.json'))
            best_valid = best[0].split('_')[-3]

            os.makedirs(os.path.join(EXPERIMENTS_DIR, f'experiments_{CLASSIFIER}', f'{data}_{metric}'), exist_ok=True)
            for i in range(0, 5):
                copyfile(os.path.join(EXPERIMENTS_DIR, f'{DATA_NAME}_{data}', f'output_{CLASSIFIER}', f'test_{metric}_{best_valid}_0_{i}.json'),
                         os.path.join(EXPERIMENTS_DIR, f'experiments_{CLASSIFIER}', f'{data}_{metric}', f'eval_preds_test_{i}.json'))

            best_models.append(f'{data}_{metric}')

with open(BEST_DIR, 'w') as outfile:
    json.dump(best_models, outfile)
