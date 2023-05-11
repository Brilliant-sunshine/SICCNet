import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne(data, label, path, subject):
    X = TSNE(n_components=2, random_state=0).fit_transform(data)
    y = label.T
    plt.style.use('seaborn-paper')

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    X_l, Y_l = X[:, 0], X[:, 1]

    ax = plt.subplot(111)
    plt.figure(facecolor="white", figsize=(3.5, 2.8))

    palette = np.array(sns.color_palette("hls", 4))

    ind_1 = np.where(y == 0)[1]
    lab1 = plt.scatter(X[ind_1, 0], X[ind_1, 1],  color='blue')#color='c',
    ind_2 = np.where(y == 1)[1]
    lab2 = plt.scatter(X[ind_2, 0], X[ind_2, 1], color='forestgreen')#color='b' ) #
    ind_3 = np.where(y == 2)[1]
    lab3 = plt.scatter(X[ind_3, 0], X[ind_3, 1],  color='r') # marker='p', color='r', s=35
    ind_4 = np.where(y == 3)[1]
    lab4 = plt.scatter(X[ind_4, 0], X[ind_4, 1],  color='orange') # , marker='*', color='y', label='3', s=35

    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)

    plt.legend(frameon=True,loc = 'lower right', handles = [lab1, lab2, lab3, lab4],
               labels = ['LH','RH', 'F', 'T'],
               prop={'family': 'Times New Roman','size': 9})

    plt.tick_params(labelsize=9)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax = plt.gca()
    ax.axes.xaxis.set_visible(True)
    ax.axes.yaxis.set_visible(True)

    plt.text(0.2, 1.3, "Subject ", fontdict={'family': "Times New Roman", 'size': 16})
    plt.text(0.6, 1.3, "A03", fontdict={'family': "Times New Roman", 'size': 16, 'fontstyle': "italic"})
    plt.savefig(path)
    plt.show()