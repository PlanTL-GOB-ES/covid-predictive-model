from evaluation.evaluate_jsons import evaluate_jsons
import os
import time
import sys
import json

METRIC = 'accuracy'
assert METRIC in ['f1', 'accuracy']

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python create_best_cv_json.py PATH_TO_EXPERIMENTS')
    print('Note: skipping directories starting with _')
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    dummy_dir = os.path.join(sys.argv[1], f'_eval_cv_json_{timestamp}')
    print('Saving recomputed eval reports in', dummy_dir, '(can be deleted)')
    os.makedirs(dummy_dir)  # can be deleted. Here eval reports etc are recomputed, in case you needed
    scores = {}
    path = sys.argv[1]
    skipped = 0
    for idx, experiment_path in enumerate(os.listdir(path)):
        if not os.path.isdir(os.path.join(path, experiment_path)) or experiment_path.startswith('_'):
            skipped += 1
        else:
            dummy_dir_inner = os.path.join(dummy_dir, experiment_path)
            os.makedirs(dummy_dir_inner, exist_ok=True)
            try:
                valid_metrics, test_metrics = evaluate_jsons([os.path.join(path, experiment_path)], 0.5, dummy_dir_inner,
                                                             metric=METRIC)
                scores[experiment_path] = valid_metrics[f'{METRIC}_avg_weighted']['mean']
                print(f'Experiment {1 + idx - skipped} of '
                      f'{len([p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p)) and not p.startswith("_")])}')
            except BaseException as e:
                skipped += 1
                print('Skipping', experiment_path, ':', e)
                scores[experiment_path] = 0.0
    with open(os.path.join(path, f'best_cv_{METRIC}.json'), 'w') as f:
        json.dump({k: v for k, v in sorted(scores.items(), reverse=True, key=lambda item: item[1])}, f)
    print('Saved json in', os.path.join(path, f'best_cv_{METRIC}.json'))
