from matplotlib import pyplot as plt

color_blue = 'cornflowerblue'
color_green = '#60A917'

xs = [0, 1, 2, 3, 4]
y = [0.9088, 0.9057, 0.9014, 0.8965, 0.8900]
y_std = [0.0127, 0.0141, 0.0155, 0.0171, 0.0187]

y_before = [0.9088, 0.9125, 0.9159, 0.9178, 0.9170]
y_before_std = [0.0127, 0.0126, 0.0127, 0.0132, 0.0132]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(xs, y_before, 'o-', color=color_blue, label="From outcome")
ax.plot(xs, y, 'o-', color=color_green, label="From admission")
ax.fill_between(range(len(xs)), [i - y_std for i, y_std in zip(y, y_std)], [i + y_std for i, y_std in zip(y, y_std)],
                alpha=.1, color=color_green)
ax.fill_between(range(len(xs)), [i - y_std for i, y_std in zip(y_before, y_before_std)],
                [i + y_std for i, y_std in zip(y_before, y_before_std)],
                alpha=.1, color=color_blue)
ax.set(xlabel='Days', ylabel='Accuracy')
ax.set_xticks([0, 1, 2, 3, 4])
ax.legend(loc='lower left')
fig.savefig('by_days.png')
