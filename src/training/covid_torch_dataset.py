from torch.utils.data import Dataset
import pandas as pd
import os
from typing import List, Tuple, Iterable
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import StratifiedShuffleSplit


class CovidData:
    def __init__(self, csvs_dir: str, static_table_name: str, dynamic_table_name: str, labels_table_name: str,
                 debug: bool = False, filter_patients_ids: List = [], features_selected: List = []):
        self.csvs_dir = csvs_dir
        self.static_table_name = static_table_name
        self.dynamic_table_name = dynamic_table_name
        self.labels_table_name = labels_table_name
        self.debug = debug

        self.static_data = pd.read_csv(os.path.join(csvs_dir, static_table_name), header=0)
        self.dynamic_table = pd.read_csv(os.path.join(csvs_dir, dynamic_table_name), header=0)
        self.labels = pd.read_csv(os.path.join(csvs_dir, labels_table_name), header=0)

        is_sorted = True
        if not self.static_data['patientid'].is_monotonic_increasing:
            is_sorted = False
            self.static_data.sort_values(by=['patientid'])
        if not self.dynamic_table['patientid'].is_monotonic_increasing:
            is_sorted = False
            self.dynamic_table.sort_values(by=['patientid'])
        if not self.labels['patientid'].is_monotonic_increasing:
            is_sorted = False
            self.labels.sort_values(by=['patientid'])

        # Currently not used. Potentially used for warning the user that sorting has been enforced
        self.was_sorted = is_sorted

        # filter patient ids direcly from dataframes
        if filter_patients_ids:
            self.static_data = self.static_data[~self.static_data['patientid'].isin(filter_patients_ids)]
            self.labels = self.labels[~self.labels['patientid'].isin(filter_patients_ids)]
            self.dynamic_table = self.dynamic_table[~self.dynamic_table['patientid'].isin(filter_patients_ids)]

        # Select features from list in static and dynamic data that intersect the selected features
        # The "patient_id" columns is kept for further use
        if features_selected:
            features_selected_static = sorted(set(features_selected).intersection(set(self.static_data.columns.to_list())))
            features_selected_dynamic = sorted(set(features_selected).intersection(set(self.dynamic_table.columns.to_list())))

            # Important the order of patientid. It should be first in list.
            # NOTE: All elements are important for the order, not only patientid!
            self.static_data = self.static_data[['patientid'] + features_selected_static]
            self.dynamic_table = self.dynamic_table[['patientid'] + features_selected_dynamic]

        self.static_data = self.static_data.values.astype(np.float32)
        self.labels = self.labels.values[:, 1]
        self.dynamic_table = self.dynamic_table.values.astype(np.float32)

        self.patient_id = list(self.static_data[:, 0])
        if debug:
            load = 100
            self.static_data = self.static_data[:load]
            self.labels = self.labels[:load]
            self.patient_id = self.patient_id[:load]
        self.dynamic_data = []
        last_index = 0
        for idx in tqdm(range(len(self.static_data))):
            idx = self.static_data[idx][0]
            data = []
            found = False
            first = True
            for index_table in range(last_index, len(self.dynamic_table)):
                row = self.dynamic_table[index_table]
                id_ = int(row[0])
                if first:
                    first = False
                    data.append(row[1:])
                    continue
                if id_ == idx:
                    found = True
                    data.append(row[1:])
                elif found:
                    last_index = index_table
                    break
            # Add empty vector for missing ids in the dynamic tables
            if len(data) == 0:
                self.dynamic_data.append([np.zeros(self.dynamic_table.shape[1] - 1, dtype=np.float32)])
            else:
                self.dynamic_data.append(data)

        self.static_data = self.static_data[:, 1:]
        self.idxs = list(range(len(self.static_data)))


class CovidDataset(Dataset):
    def __init__(self, data: CovidData, subset: str = None, seed: int = 42, props: Tuple[float] = (80, 10, 10),
                 stratified: bool = False, cv: int = 1, nested_cv: int = 1, cv_idx: int = 0, nested_cv_idx: int = 0,
                 debug: bool = False, seed_cv: int = 42):
        # seed should NOT be used here (TODO review)
        super().__init__()
        subsets = ['train', 'valid', 'test']
        #assert subset in subsets
        assert sum(props) == 100
        self.data = data
        self.props = props
        self.proportions = dict(zip(subsets, self.props))
        self.subset = subset
        self.seed = seed
        self.seed_cv = seed_cv
        self.stratified = stratified
        self.cv = cv
        self.nested_cv = nested_cv
        self.cv_idx = cv_idx
        self.nested_cv_idx = nested_cv_idx
        self.debug = debug

        self.static_data = self.data.static_data
        self.labels = self.data.labels
        self.patient_id = self.data.patient_id
        self.dynamic_data = self.data.dynamic_data
        # np.random.seed(self.seed)
        self.idxs = self.data.idxs

        if self.stratified:
            sss_outer = StratifiedShuffleSplit(n_splits=self.nested_cv, test_size=self.proportions['test'] / 100,
                                               random_state=self.seed)
            learn_idx, test_idx = list(sss_outer.split(self.idxs, self.labels))[self.nested_cv_idx]

            sss_inner = StratifiedShuffleSplit(n_splits=self.cv,
                                               test_size=(self.proportions['valid'] / (self.proportions['train'] +
                                                                                       self.proportions['valid'])),
                                               random_state=self.seed_cv)
            learn = np.array(self.idxs)[learn_idx]
            learn_labels = self.labels[learn_idx]
            train_idx, valid_idx = list(sss_inner.split(learn, learn_labels))[self.cv_idx]
            train_idx = learn[train_idx]
            valid_idx = learn[valid_idx]
            if self.subset == 'train':
                self.idxs = list(train_idx)
            elif self.subset == 'valid':
                self.idxs = list(valid_idx)
            else:
                self.idxs = list(test_idx)
        else:
            if cv != 1 or nested_cv != 1:
                raise NotImplementedError('CV is not implemented without stratified split')
            if seed_cv != seed:
                raise NotImplementedError('Different seed for CV is not implemented without stratified split')
            np.random.shuffle(self.idxs)
            if self.subset:
                self.idxs = [x for idx, x in enumerate(self.idxs) if self._check_idx(idx + 1, self.subset)]
        self.patient_id = self.patient_id
        self.patient_id = [self.patient_id[id_] for id_ in self.idxs]
        self.static_data = self.static_data[self.idxs]
        self.dynamic_data = [self.dynamic_data[id_] for id_ in self.idxs]
        self.labels = self.labels[self.idxs]
        self.lengths = [len(seq) for seq in self.dynamic_data]

    def __getitem__(self, index: int) -> Tuple[int, np.ndarray, List, int, List[int]]:
        return self.patient_id[index], self.static_data[index], self.dynamic_data[index], self.lengths[index], \
               self.labels[index]

    def __len__(self) -> int:
        return len(self.static_data)

    def _check_idx(self, i: int, sub: str) -> bool:
        if sub == 'train' and i % 100 < self.proportions['train']:
            return True
        elif sub == 'valid' and self.proportions['train'] <= i % 100 < (self.proportions['train'] +
                                                                        self.proportions['valid']):
            return True
        elif sub == 'test' and i % 100 >= (self.proportions['train'] + self.proportions['valid']):
            return True
        return False

    def get_label(self, dt, idx):
        return self.__getitem__(idx)[4]

    @staticmethod
    def collate(data: List) -> Tuple[List[int], torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        ids = [item[0] for item in data]
        static_data = torch.tensor([item[1] for item in data])
        dynamic_data = [torch.tensor(item[2]) for item in data]
        lengths = torch.tensor([item[3] for item in data])
        labels = torch.tensor([item[4] for item in data])
        return ids, static_data, dynamic_data, lengths, labels


class DatasetBuilder:
    def __init__(self, data: CovidData, seed: int = 42, seed_cv: int = 15):
        self.data = data
        self.seed = seed
        self.seed_cv = seed_cv

    def build_datasets(self, cv: int, nested_cv: int) -> Iterable[
        Tuple[Iterable[Tuple[CovidDataset, CovidDataset]], CovidDataset]]:
        seed = self.seed
        seed_cv = self.seed_cv
        for i in range(nested_cv):
            def _inner_loop(data: CovidData, seed: int, seed_cv: seed):
                for j in range(cv):
                    yield (CovidDataset(data=data, subset='train', stratified=True, cv=cv, nested_cv=nested_cv,
                                        cv_idx=j, nested_cv_idx=i, seed=seed, seed_cv=seed_cv),
                           CovidDataset(data=data, subset='valid', stratified=True, cv=cv, nested_cv=nested_cv,
                                        cv_idx=j, nested_cv_idx=i, seed=seed, seed_cv=seed_cv))

            yield (_inner_loop(self.data, self.seed, self.seed_cv),
                   CovidDataset(data=self.data, subset='test', stratified=True, cv=cv, nested_cv=nested_cv, cv_idx=0,
                                nested_cv_idx=i, seed=seed, seed_cv=seed_cv))


if __name__ == '__main__':
    # Example usage
    csvs_dir = os.path.join('..', 'db', 'datasets')
    static_table = 'dataset_static.csv'
    dynamic_table = 'dataset_dynamic.csv'
    labels_table = 'dataset_labels.csv'
    data = CovidData(csvs_dir=csvs_dir, static_table_name=static_table, dynamic_table_name=dynamic_table,
                     labels_table_name=labels_table, debug=False)

    subset = 'valid'
    dataset = CovidDataset(data=data, subset=subset, stratified=True)
    dl = DataLoader(dataset=dataset, shuffle=(subset == 'train'), batch_size=2, collate_fn=dataset.collate)
    for batch in dl:
        print(batch)
        break

    # CV
    dataset = CovidDataset(data=data, subset=subset, stratified=True, cv=5, cv_idx=3)
    dl = DataLoader(dataset=dataset, shuffle=(subset == 'train'), batch_size=2, collate_fn=dataset.collate)
    for batch in dl:
        print(batch)
        break

    # (Nested) CV with DatasetBuilder
    db = DatasetBuilder(data=data, seed=42)
    cv = 5
    nested_cv = 3
    # Outer loop
    for inner_train_valids, test in db.build_datasets(cv=cv, nested_cv=nested_cv):
        test_ids = set(test.patient_id)
        assert len(test_ids) == len(test)
        for train, valid in inner_train_valids:
            print(len(train), len(valid), len(test))
            train_ids = set(train.patient_id)
            assert len(train_ids) == len(train)
            train_dl = DataLoader(dataset=train, shuffle=True, batch_size=2, collate_fn=dataset.collate)
            for batch in train_dl:
                break
            valid_ids = set(valid.patient_id)
            assert len(valid_ids) == len(valid)
            valid_dl = DataLoader(dataset=valid, shuffle=False, batch_size=2, collate_fn=dataset.collate)
            for batch in valid_dl:
                break
            assert len(train_ids.intersection(valid_ids)) == 0 and len(train_ids.intersection(test_ids)) == 0 and \
                   len(valid_ids.intersection(test_ids)) == 0
        test_dl = DataLoader(dataset=test, shuffle=False, batch_size=2, collate_fn=dataset.collate)

        for batch in test_dl:
            break

        print()
