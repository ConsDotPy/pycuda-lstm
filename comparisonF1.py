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

labels = ['Indexado Frecuencia', 'Word2Vec', 'GloVe']

men_means = [0.681, 0.591, 0.55]
women_means = [0.827, 0.859, 0.89]
tc = [0.6842, 0.713, 0.795]

x = np.arange(0, 4*len(labels), 4)    # the x locations for the groups

width = 0.80  # the width of the bars

fig, ax = plt.subplots()
ax.set_xticks(x + 3*width)
rects1 = ax.bar(x - width, men_means, width, label='Loss', align='center')
rects2 = ax.bar(x, women_means, width, label='Accuracy', align='center')
rects3 = ax.bar(x + width, tc, width, label='F1 - Score', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Valor')
ax.set_title('Valores para Embebidos de Palabras')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()