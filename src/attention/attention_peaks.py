import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import manifold, cluster, tree, model_selection
from data_analysis.extract_features import extract_features_information_metrics
from matplotlib import pyplot as plt, cm, colors
from scipy.cluster import hierarchy

PATH_TO_DATA = ""
PATH_TO_LABELS = ""
PATH_TO_CSV = ""
PATH_TO_FEATURES = None
PATH_TO_PATIENT_SELECTION = ""
DAY_OFFSET = 0
N_FEATURES = 150
DEATH = True


def lookup_data(df, x):
    patient_days = df[df["patientid"] == x['patientid']]

    for idx, (_, patient_day) in enumerate(patient_days.iterrows()):
        if (idx + DAY_OFFSET) == x["peak_day"]:
            patient_day["label"] = 1 if x.get('significant', 0) else 0
            return patient_day


def lookup_all(df, x):
    data = list()
    cur_patient = None
    cur_patient_idx = None
    for idx, (_, patient_day) in enumerate(df.iterrows()):
        if cur_patient and cur_patient == patient_day['patientid']:
            cur_patient_idx = cur_patient_idx + 1
        else:
            cur_patient = patient_day['patientid']
            cur_patient_idx = 1

        d = x[(x['patientid'] == cur_patient) & (x['peak_day'] == (cur_patient_idx + DAY_OFFSET)) & x["significant"]]
        label = 0 if d.empty else 1
        patient_day['label'] = label
        data.append(patient_day)
    return pd.DataFrame(data)


def shuffle_split(df):
    df = df.sample(frac=1)
    df1 = df[df['label'] == 1]
    df2 = df[df['label'] == 0].iloc[0:df1.shape[0] * 3]
    df = pd.concat((df1, df2))
    return df[[col for col in df if col != 'patientid' and col != 'label']], df['label']


def cluster_peaks(df, name, idx):
    plt.figure(figsize=(10, 7))
    plt.title("Dendograms")
    dend = hierarchy.dendrogram(hierarchy.linkage(df, method='ward'))
    plt.savefig(os.path.join(PATH_TO_CSV, f'dendogram_{name}_{idx}.png'))
    '''
    model = clustering.AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    labels = model.fit_predict(df)
    '''
    model = cluster.KMeans(2)
    labels = model.fit_predict(df)
    # print("Number of clusters found: ", str(len(set(labels))))
    return labels


def classify_decision_tree(df, label, name, idx):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(df, label, test_size=0.15)

    model = tree.DecisionTreeClassifier(max_depth=7, min_samples_leaf=5, class_weight='balanced')
    model.fit(x_train, y_train)
    print("Decision Tree Accuracy: ", model.score(x_test, y_test))
    plt.figure(figsize=(60, 60))
    filename = os.path.join(PATH_TO_CSV, f'dt_{name}_{idx}')
    tree.export_graphviz(model, out_file=f'{filename}.dot', feature_names=df.columns, class_names=True)
    os.system(f'dot -Tpng {filename}.dot > {filename}.png')


def show_manifold(df, labels, name, idx):
    cmap = cm.get_cmap('Accent')
    norm = colors.Normalize(vmin=labels.min(), vmax=labels.max())
    c = [cmap(norm(label)) for label in labels]
    for method in [manifold.TSNE]:
        embedded = method(n_components=2).fit_transform(df)
        plt.figure(figsize=(15, 15), dpi=300)
        plt.scatter(embedded[:, 0], embedded[:, 1], c=c)
        plt.title(method.__name__)
        plt.savefig(os.path.join(PATH_TO_CSV, f'manifold_{method.__name__}_{name}_{idx}.png'))


if __name__ == '__main__':
    # Load
    labels = pd.read_csv(PATH_TO_LABELS)
    #features = extract_features_information_metrics(PATH_TO_FEATURES, N_FEATURES)
    #features.append('patientid')
    df_dynamic = pd.read_csv(PATH_TO_DATA)
    features = df_dynamic.columns  #if col in features]
    df_dynamic = df_dynamic[features]
    if DEATH:
        df_dynamic = df_dynamic.merge(labels, how='left', on='patientid')

    with open(PATH_TO_PATIENT_SELECTION) as f:
        patients = json.load(f)['test']
    df_dynamic = df_dynamic[df_dynamic['patientid'].isin(patients)]

    ratios_peaks = []
    ratios_patients_peaks = []
    for i in tqdm(range(5)):
        df_peaks = pd.read_csv(os.path.join(PATH_TO_CSV, f'peaks_{i}.csv'))
        df_peaks_no = pd.read_csv(os.path.join(PATH_TO_CSV, f'no_peaks_{i}.csv'))
        df_dynamic_embedding = pd.read_csv(os.path.join(PATH_TO_CSV, f'dynamic_embeddings_{i}.csv'))
        df_peaks = df_peaks[df_peaks['patientid'].isin(patients)]
        df_peaks_no = df_peaks_no[df_peaks_no['patientid'].isin(patients)]
        df_all = lookup_all(df_dynamic, df_peaks)

        # Filter
        n_peaks = len(df_peaks)
        df_peaks = df_peaks[df_peaks['significant']]
        ratios_peaks.append(len(df_peaks)/n_peaks)
        ratios_patients_peaks.append(len(df_peaks['patientid'].unique())/len(patients))
        # Lookup
        df_peaks = df_peaks.apply(lambda x: lookup_data(df_dynamic, x), axis=1)
        col_filter = [col for col in df_peaks.columns if col != 'label' and col != 'patientid']
        ##
        df_peaks_no = df_dynamic[df_dynamic['patientid'].isin(df_peaks_no['patientid'])]
        df_peaks_no['label'] = 0
        # --------------------------------
        """
        print("Clusterize peaks")
        embedding_col_filter = [col for col in df_dynamic_embedding.columns if col not in ['0', '1']]
        embedding_cluster_labels = cluster_peaks(df_dynamic_embedding[embedding_col_filter], 'dyn_emb', idx=i)
        cluster_labels = cluster_peaks(df_peaks[col_filter], 'peaks', idx=i)
        # --------------------------------
        print("Classify as Decision Tree")
        classify_decision_tree(df_dynamic_embedding[embedding_col_filter], embedding_cluster_labels,
                               name='embedding_cluster', idx=i)
        classify_decision_tree(df_dynamic_embedding[embedding_col_filter], df_dynamic_embedding['1'],
                               name='embedding', idx=i)
        classify_decision_tree(df_peaks[col_filter], cluster_labels, name='only' + ('_death' if DEATH else ''), idx=i)
        df_all, df_all_labels = shuffle_split(df_all)
        classify_decision_tree(df_all, df_all_labels, name='all' + ('_death' if DEATH else ''), idx=i)
        # --------------------------------
        print("Show peaks in lower dimensional space")
        show_manifold(df_dynamic_embedding[embedding_col_filter], df_dynamic_embedding['1'], 'emb_emb', idx=i)
        show_manifold(df_dynamic_embedding[embedding_col_filter], embedding_cluster_labels, 'emb_clu_labels', idx=i)
        show_manifold(df_all, df_all_labels, 'all_labels', idx=i)
        """
    # --------------------------------
    print('Statistics')
    print(np.mean(ratios_peaks), np.std(ratios_peaks))
    print(np.mean(ratios_patients_peaks), np.std(ratios_patients_peaks))
