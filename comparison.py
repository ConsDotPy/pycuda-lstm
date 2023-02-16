import matplotlib
import matplotlib.pyplot as plt
import numpy as np

timesteps = 19

labels = ['128 - 5000', '256 - 5000', '512 - 5000']
a = (2*314311850000.0)/(675.42*1e9)
b = (2*1694719560000.0)/(2246.88*1e9)
c = (2*10678181975000.0)/(8748.37*1e9)

cpu = [675.42, 2246.88, 8748.37]
forward = [529.80, 1500.24, 5667.96]
last = [474.36, 1292.06, 4937.65]
now = [474.36 - 90, 1292.06 - 180, 4937.65 - 850]

men_means = [round((2*314311850000.0)/(cpu[0]*1e9), 2), round((2*1694719560000.0)/(cpu[1]*1e9), 3), round((2*10678181975000.0)/(cpu[2]*1e9), 3)]
women_means = [round((2*314311850000.0)/(forward[0]*1e9), 2), round((2*1694719560000.0)/(forward[1]*1e9), 3), round((2*10678181975000.0)/(forward[2]*1e9), 3)]
tc = [round((2*314311850000.0)/(last[0]*1e9), 3), round((2*1694719560000.0)/(last[1]*1e9), 3), round((2*10678181975000.0)/(last[2]*1e9), 3)]
tc2 = [round((2*314311850000.0)/(now[0]*1e9), 3), round((2*1694719560000.0)/(now[1]*1e9), 3), round((2*10678181975000.0)/(now[2]*1e9), 3)]

x = np.arange(0, 4*len(labels), 4)    # the x locations for the groups

width = 0.80  # the width of the bars

fig, ax = plt.subplots()
ax.set_xticks(x + 3*width)
rects1 = ax.bar(x - width, men_means, width, label='CPU', align='center')
rects2 = ax.bar(x, women_means, width, label='GPU - F', align='center')
rects3 = ax.bar(x + width, tc, width, label='GPU - FB', align='center')
rects4 = ax.bar(x + 2*width, tc2, width, label='GPU - C', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Batch/s')
ax.set_title('Batches per second - Hidden = Batch')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.show()

labels = ['128 - 5000', '256 - 5000', '512 - 5000']
men_means = [round((2*314311850000.0)/(a*cpu[0]*1e9), 2), round((2*1694719560000.0)/(b*cpu[1]*1e9), 3), round((2*10678181975000.0)/(c*cpu[2]*1e9), 3)]
women_means = [round((2*314311850000.0)/(a*forward[0]*1e9), 2), round((2*1694719560000.0)/(b*forward[1]*1e9), 3), round((2*10678181975000.0)/(c*forward[2]*1e9), 3)]
tc = [round((2*314311850000.0)/(a*last[0]*1e9), 2), round((2*1694719560000.0)/(b*last[1]*1e9), 3), round((2*10678181975000.0)/(c*last[2]*1e9), 3)]
tc2 = [round((2*314311850000.0)/(a*now[0]*1e9), 2), round((2*1694719560000.0)/(b*now[1]*1e9), 3), round((2*10678181975000.0)/(c*now[2]*1e9), 3)]

x = np.arange(0, 4*len(labels), 4)    # the x locations for the groups

width = 0.80  # the width of the bars

fig, ax = plt.subplots()
ax.set_xticks(x + 2*width)
rects1 = ax.bar(x - width, men_means, width, label='CPU', align='center')
rects2 = ax.bar(x, women_means, width, label='GPU - F', align='center')
rects3 = ax.bar(x + width, tc, width, label='GPU - FB', align='center')
rects4 = ax.bar(x + 2*width, tc2, width, label='GPU - C', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Coeficiente de aceleración')
ax.set_title('Aceleración respecto a Batch/s')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.show()