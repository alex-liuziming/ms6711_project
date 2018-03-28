
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewide=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')