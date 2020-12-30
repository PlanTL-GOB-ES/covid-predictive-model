import os
import csv
from statistics import mean, stdev
import numpy as np

FOLDER = ""
SCORES_PATHS = []
'''
Makes a mean of the scores.csv of the selected models.
'''
metrics = list()
for pathh in SCORES_PATHS:
    file = os.path.join(FOLDER, pathh, "scores.csv")
    reader = csv.reader(open(file, "r"), delimiter=",")
    scores = list(reader)[1:]
    scores = np.array(scores).astype("float")
    metrics.append([(mean([scores[i][j] for i in range(len(scores))]),
                     stdev([scores[i][j] for i in range(len(scores))]))
                    for j in range(1, 5)])

output = [(mean([metrics[i][j][0] for i in range(len(metrics))]), mean([metrics[i][j][1] for i in range(len(metrics))]))
       for j in range(4)]

print(output)

output = ' & '.join([f'{o[0]:.4f}$\\pm${o[1]:.4f}' for o in output])

print(output)


