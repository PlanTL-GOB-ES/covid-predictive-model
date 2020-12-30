import os
import fnmatch
import pandas as pd


def extract_features_random_forest(feature_dir, rank=None):
    types = ['dynamic', 'static']
    features = dict()
    for _type in types:
        features[_type] = set()
        first = True
        for filename in os.listdir(feature_dir):
            if fnmatch.fnmatch(filename, f'*{_type}*'):
                with open(os.path.join(feature_dir, filename)) as fn:
                    feats = [line.split('\t')[0] for line in fn.readlines()
                             if line != 'patientid']
                    if rank:
                        feats = feats[:rank]
                if first:
                    features[_type] = set(feats)
                else:
                    features[_type].intersection(set(feats))
                first = False
        features[_type] = list(features[_type])

    return features


def extract_features_information_metrics(feature_path, top_n=None):
    df = pd.read_csv(feature_path, sep='\t', encoding='latin')

    features = []
    for col in [col for col in df.columns if col != 'feature']:
        if top_n:
            feats = df.iloc[df[col].sort_values(ascending=False)[:top_n].index]['feature'].to_list()
        else:
            feats = df.iloc[df[col].sort_values(ascending=False).index]['feature'].to_list()
        features.extend(feats)

    features_common = list(set(features))
    return features_common


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # feature_dir = os.path.join(script_dir, 'feature_importance', 'random_forest')
    # features = extract_features_random_forest(feature_dir, rank=100)
    feature_path = os.path.join(script_dir, 'feature_importance', 'information_metrics', 'feature_importance.tsv')
    features = extract_features_information_metrics(feature_path, top_n=300)
    df = pd.read_csv(feature_path, sep='\t')
    print(features)
