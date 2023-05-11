#coding=utf8
import numpy as np
import itertools
import matplotlib.pyplot as plt

NUM_CLASSES = 4
CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis',
                'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']
TARGET_NAME = ['LH', 'RH', 'F', 'T']

def plot_confusion_matrix(epoch,
                          cm,
                          fold,
                          dataset,
                          subject,
                          target_names=None,
                          title='A02',
                          cmap=None,
                          normalize=True):

    if target_names is None:
        target_names = TARGET_NAME
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm = cm*100

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    figure, ax = plt.subplots(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title("Subject ", fontdict={'family': "Times New Roman", 'size': 18})
    # plt.title(title, fontdict={'family':"Times New Roman", 'size':18, 'fontstyle':"italic"})
    plt.colorbar(fraction=0.046)
    plt.clim(0,100)
    plt.text(0.75,-0.7, "Subject ", fontdict={'family':"Times New Roman", 'size':18})
    plt.text(1.75, -0.7, "B01", fontdict={'family': "Times New Roman", 'size': 18, 'fontstyle':"italic"})

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    cm_ori = cm

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="black",
                 fontdict={'family':"Times New Roman", 'size':16 })
        # plt.text(j, i, "{:}\n{:0.2f}%".format(cm_ori[i, j], cm[i, j] * 100),
        #          horizontalalignment="center",
        #          color="black")

    plt.tick_params(labelsize=16)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.tight_layout()
    plt.ylabel('True label', fontdict={'family':"Times New Roman", 'size':18})
    plt.xlabel('Predicted label', fontdict={'family':"Times New Roman", 'size':18})

    filename = 'confusion_matrix_epo'+str(epoch)+'_fold'+str(fold)+'S'+str(subject)+'_new3.svg'
    folder = './images/confusion_matrix/'+str(dataset)+'-Shallow/S'+str(subject)+'/'
    ch_filepath = folder + filename
    plt.savefig(ch_filepath, dpi=200, bbox_inches='tight')
    plt.show()

