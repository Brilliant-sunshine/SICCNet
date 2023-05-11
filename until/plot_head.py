import numpy as np
import mne
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.datasets.sensor_positions import get_channelpos, CHANNEL_10_20_APPROX
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_head_show(train_set, map_avg, map2, subject_id, fold, after_n):

    iterator = BalancedBatchSizeIterator(batch_size=40)  # 迭代器
    train_batches = list(iterator.get_batches(train_set, shuffle=False))
    train_X_batches = np.concatenate(list(zip(*train_batches))[0])

    map_mean = np.mean(map_avg, axis=0)
    map_mean2 = np.mean(map2, axis=0)

    fs = 250
    freqs = np.fft.rfftfreq(train_X_batches.shape[2], d=1.0 / fs)  # 返回离散傅里叶变换采样频率

    start_freq = 4
    stop_freq = 38

    i_start = np.searchsorted(freqs, start_freq)
    i_stop = np.searchsorted(freqs, stop_freq) + 1

    freq_corr = np.mean(map_mean[:, i_start:i_stop], axis=1)
    freq_corr2 = np.mean(map_mean2[:, i_start:i_stop], axis=1)

    ch_names = ['Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'C5..', 'C3..', 'C1..', 'Cz..',
                'C2..', 'C4..', 'C6..', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Fz..',
                'P1..', 'Pz..', 'P2..', 'Poz.']

    ch_names = [s.strip('.') for s in ch_names]
    positions = [get_channelpos(name, CHANNEL_10_20_APPROX) for name in ch_names]
    positions = np.array(positions)

    max_abs_val = np.max(np.abs(freq_corr))
    max_abs_val2 = np.max(np.abs(freq_corr2))

    max = max_abs_val if max_abs_val>max_abs_val2 else max_abs_val2

    plt.figure(figsize=(3, 12))
    fig, axes = plt.subplots(2, 4)
    ax = axes.flatten()
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    for i_class in range(4):
        ax = axes[0, i_class]
        im, cn = mne.viz.plot_topomap(freq_corr[:, i_class], positions, res=64 * 64,
                             vmin=-max, vmax=max, contours=0,
                             cmap=cm.coolwarm, axes=ax, show=False)
        ax.set_title(class_names[i_class])

    for i_class in range(4):
        ax = axes[1, i_class]
        im1, cn = mne.viz.plot_topomap(freq_corr2[:, i_class], positions, res=64 * 64,
                             vmin=-max, vmax=max, contours=0,
                             cmap=cm.coolwarm, axes=ax, show=False)
        ax.set_title(class_names[i_class])

    plt.text(-5.5, -1, "min", fontdict={'size': 14})
    plt.text(-5.5, 0, "max", fontdict={'size': 14})
    plt.text(-5.5, 1, "0", fontdict={'size': 14})

    position = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    # cbar = plt.colorbar(im, fraction=0.05, pad=0.05, cax=position, orientation='horizontal')
    # cbar1 = plt.colorbar(im1, fraction=0.05, pad=0.05, cax=position, orientation='horizontal', )
    cbar1 = plt.colorbar(im1, ticks=range(0), cax=position, orientation='horizontal')
    cbar1.set_label('Correlation coefficient')
    # cbar.set_ticks(['min', '0', 'max'])
    # cbar.set_ticklabels(['min', '0', 'max'])
    path = "./images/map/AllSubject_new/"
    plt.savefig(path + "/all_new_f1_NEW_.svg", format="svg", bbox_inches='tight')
    plt.show()

class data():
    def __init__(self, data, label):
        self.X = data
        self.y = label