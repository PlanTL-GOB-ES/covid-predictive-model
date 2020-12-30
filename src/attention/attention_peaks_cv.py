import os
import json
from tqdm import tqdm
import pandas as pd
from sklearn import tree, model_selection
from data_analysis.extract_features import extract_features_information_metrics
from matplotlib import pyplot as plt
from sklearn import metrics

PATH_TO_DATA = ""
PATH_TO_LABELS = ""
PATH_TO_CSV = ""
PATH_TO_FEATURES = None 
PATH_TO_PATIENT_SELECTION = ""
DAY_OFFSET = 0
N_FEATURES = 150


def lookup_all(df, x):
    data = list()
    cur_patient = None
    cur_patient_idx = None
    for idx, (_, patient_day) in enumerate(df.iterrows()):
        if cur_patient and (cur_patient == patient_day['patientid']):
            cur_patient_idx = cur_patient_idx + 1
        else:
            cur_patient = patient_day['patientid']
            cur_patient_idx = 1

        d = x[(x['patientid'] == cur_patient) & (x['peak_day'] == (cur_patient_idx + DAY_OFFSET)) & x["significant"]]
        patient_day['peak'] = not d.empty
        data.append(patient_day)
    return pd.DataFrame(data)


def shuffle_split(df):
    df = df.sample(frac=1)
    df1 = df[df['label'] == 1]
    df2 = df[df['label'] == 0].iloc[0:df1.shape[0] * 3]
    df = pd.concat((df1, df2))
    return df[[col for col in df if col != 'patientid' and col != 'label']], df['label']


def classify_decision_tree(df, label, name, idx):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(df, label, test_size=0.15)

    model = tree.DecisionTreeClassifier(class_weight='balanced')
    model.fit(x_train, y_train)
    y_test_preds = model.predict(x_test)
    print(metrics.classification_report(y_test, y_test_preds))
    print(metrics.f1_score(y_test, y_test_preds, average='macro'))
    plt.figure(figsize=(60, 60))
    filename = os.path.join(PATH_TO_CSV, f'dt_{name}_{idx}')
    tree.export_graphviz(model, out_file=f'{filename}.dot', feature_names=df.columns, class_names=True)
    os.system(f'dot -Tpng {filename}.dot > {filename}.png')


if __name__ == '__main__':
    # Load
    labels = pd.read_csv(PATH_TO_LABELS)
    #features = extract_features_information_metrics(PATH_TO_FEATURES, N_FEATURES)
    #features.append('patientid')
    df_dynamic = pd.read_csv(PATH_TO_DATA)
    features = df_dynamic.columns  #if col in features]
    df_dynamic = df_dynamic[features]
    df_dynamic = df_dynamic.merge(labels, how='left', on='patientid')

    with open(PATH_TO_PATIENT_SELECTION) as f:
        patients = json.load(f)['test']
    df_dynamic = df_dynamic[df_dynamic['patientid'].isin(patients)]

    for i in tqdm(range(5)):
        df_peaks = pd.read_csv(os.path.join(PATH_TO_CSV, f'peaks_{i}.csv'))
        df_peaks_no = pd.read_csv(os.path.join(PATH_TO_CSV, f'no_peaks_{i}.csv'))

        df_peaks = df_peaks[df_peaks['patientid'].isin(patients)]
        df_all = lookup_all(df_dynamic, df_peaks)

        df_all = df_all.loc[:, df_all.columns != 'patientid']
        label_columns = ['peak', 'fallecimiento']

        def labelize(x):
            if x['fallecimiento']:
                return x['peak']
            else:
                return x['peak']+2

        '''
        vivo_no_pico 0
        vivo_pico 1
        muerto_no_pico 2
        muerto_pico_3
        '''

        df_all_labels = df_all[label_columns].apply(lambda x: labelize(x), axis=1)
        df_all = df_all[[column for column in df_all.columns if column not in label_columns]]

        classify_decision_tree(df_all, df_all_labels, name='dt_4_class', idx=i)
