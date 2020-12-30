import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("./pyramid_data_h12o.csv")
sns.set_style("ticks")

bins_order = list(reversed(data['Bin']))
#data["Male"] = data["Male"] - data["ICU male"] - data["Death male"]
#data["Female"] = data["Female"] - data["ICU female"] - data["Death female"]
data["Female"] = data["Female"].apply(lambda x: -x)
#data["ICU female"] = data["ICU female"].apply(lambda x: -x)
data["Death female"] = data["Death female"].apply(lambda x: -x)

color_normal = "#60A917"
color_ICU = "#FFA809"
color_death = "#D62728"


bar_plot = sns.barplot(x="Male", y="Bin", color=color_normal, label="Total", data=data,
                       order=bins_order, lw=0.7)

bar_plot = sns.barplot(x="Female", y="Bin", color=color_normal, data=data,
                       order=bins_order, lw=0.7)

bar_plot = sns.barplot(x="Death male", y="Bin", color=color_death, alpha=0.8, label="Death", data=data,
                       order=bins_order, lw=0.4)

bar_plot = sns.barplot(x="Death female", y="Bin", color=color_death, alpha=0.8, data=data,
                       order=bins_order, lw=0.4)

plt.text(x=60, y=0.5, s="Male")
plt.text(x=-90, y=0.5, s="Female")
bar_plot.set(xlabel="Population", ylabel="Age Bin")
ticks = [-150, -100, -50, 0, 50, 100, 150, 200]
plt.xticks(ticks=ticks, labels=map(str, map(abs, ticks)))
plt.legend()
plt.show()
