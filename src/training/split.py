import os
import argparse
import numpy as np
import json
from tqdm import tqdm


def parse_field(typee, field):
    if typee != bool:
        return typee(field)
    else:
        return field == 'True'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config')
    parser.add_argument('output_path', type=str, help='Path to output')
    parser.add_argument('--nsplits', type=int, help='Number of splits', default=3)
    parser.add_argument('--type', type=str, help='Output structure: default, list_dict', default='default')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    assert args.type in ['default', 'list_dict']
    with open(args.config_path) as f:
        data = json.load(f)
    configs = list()
    types = list()
    for field in data.keys():
        configs.append(data[field])
        types.append(type(data[field][0]))

    splits = np.array(np.meshgrid(*configs)).T.reshape(-1, len(configs))
    splits = [[parse_field(types[ss_idx], ss) for ss_idx, ss in enumerate(s)] for s in splits]
    keys = list(data.keys())

    # Cleaning duplicates from attention fields when use_attention is False

    if 'use_attention' in keys:
        attention_key_n = keys.index('use_attention')
        attention_field_key_n = keys.index('attention_fields')
        delete_candidates = list()
        for split_idx, split in tqdm(enumerate(splits)):
            if not split[attention_key_n]:
                data = split[:attention_field_key_n] + split[attention_field_key_n+1:]
                for split_idx2, split2 in enumerate(splits):
                    if not split2[attention_key_n] and split_idx != split_idx2 and \
                        data == split2[:attention_field_key_n] + split2[attention_field_key_n+1:] and \
                            split_idx not in delete_candidates:
                        delete_candidates.append(split_idx2)

        for index in sorted(delete_candidates, reverse=True):
            del splits[index]

    if 'criterion' in keys:
        attention_key_n = keys.index('criterion')
        attention_field_key_n = keys.index('focal_loss_gamma')
        delete_candidates = list()
        for split_idx, split in tqdm(enumerate(splits)):
            if split[attention_key_n] == 'bce':
                data = split[:attention_field_key_n] + split[attention_field_key_n+1:]
                for split_idx2, split2 in enumerate(splits):
                    if split[attention_key_n] == 'bce' and split_idx != split_idx2 and \
                        data == split2[:attention_field_key_n] + split2[attention_field_key_n+1:]\
                            and split_idx not in delete_candidates:
                        delete_candidates.append(split_idx2)

        for index in sorted(delete_candidates, reverse=True):
            del splits[index]

    # Splitting and saving
    splits = np.array_split(splits, args.nsplits)
    if args.type == 'list_dict':
        for split_idx, split in enumerate(splits):
            split_data = list()
            for split_element in split:
                record = {keys[field_idx]: parse_field(types[field_idx], field_data) for field_idx, field_data in
                          enumerate(split_element)}
                split_data.append(record)

            with open(os.path.join(args.output_path, 'split_' + str(split_idx) + '.json'), 'w')\
                    as split_file:
                json.dump(split_data, split_file)
    elif args.type == 'default':
        for split_idx, split in enumerate(splits):
            split_data = dict()
            for split_element in split:
                for field_idx, field_data in enumerate(split_element):
                    already = split_data.get(keys[field_idx], list())
                    already.append(parse_field(types[field_idx], field_data))
                    split_data[keys[field_idx]] = already
            with open(os.path.join(args.output_path, 'split_' + str(split_idx) + '.json'),
                      'w') as split_file:
                json.dump(split_data, split_file)
