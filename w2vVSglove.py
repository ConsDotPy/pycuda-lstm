from matplotlib import pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax3 = ax1.twiny()

# ax1 iters
ax1.set_xlabel("Iteraciones")
ax1.set_xticks([1, 10, 20, 30, 50])
ax1.set_xticklabels(["10", "15", "20", '25', '30'])
ax1.tick_params(axis='x', colors='red')
# ax2 tiempo
ax2.set_xlabel("Tiempo (s)")
ax2.set_xticklabels(["5", "8", "11", '14', '17'])
ax2.set_xticks([5, 8, 11, 14, 17])
# ax3 neg
ax3.xaxis.set_ticks_position('bottom')
ax3.xaxis.set_label_position('bottom')
ax3.spines['bottom'].set_position(('outward', 36))
ax3.set_xlabel('Muestras negativas (CBOW)')
ax3.set_xticks([1, 10/2.2, 15/2.2, 20/2.2, 25])
ax3.set_xticklabels(["10", "15", "20", '25', '30'])
ax3.tick_params(axis='x', colors='green')

y1 = [19.24, 23.24, 25.25, 27.26, 27.27]  # iteraciones , 27.24,, 27.24,

x1 = [1, 10, 20, 30, 50]  # 0.2542, 0.2724

y2 = [3.09, 6.55, 7.44, 8.76, 8.76]  # 0.0576,

x2 = [1, 10/2.2, 15/2.2, 20/2.2, 25/2.2]

ax1.plot(x1, y1, 'r')
ax3.plot(x2, y2, 'g')
# Don't mess with the limits!
plt.autoscale(False)
plt.legend([""])
plt.legend(["W2V"])

plt.show()
