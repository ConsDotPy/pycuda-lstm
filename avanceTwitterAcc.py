import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


timesteps = 19

labels = ['Análisis de sentimientos(Twitter)']

men_means = [00.55]
women_means = [0.89]
tc = [0.795]

x = np.arange(0, len(labels), 1)    # the x locations for the groups

width = 0.30  # the width of the bars

fig, ax = plt.subplots()
ax.set_xticks(x + width)
rects1 = ax.bar(x - width, men_means, width, label='Loss', align='center')
rects2 = ax.bar(x, women_means, width, label='Accuracy', align='center')
rects3 = ax.bar(x + width, tc, width, label='F1 - Score', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Valor')
ax.set_title('Valores de inferencia del modelo')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()