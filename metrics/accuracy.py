import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from visualize import random_colors

sns.set_style('darkgrid')
plt.figure(figsize=[16, 9])
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 2.0

classes = np.asarray(['aeroplane',
                      'bicycle',
                      'bird',
                      'boat',
                      'bottle',
                      'bus',
                      'car',
                      'cat',
                      'chair',
                      'cow',
                      'diningtable',
                      'dog',
                      'horse',
                      'motorbike',
                      'person',
                      'potted plant',
                      'sheep',
                      'sofa',
                      'train',
                      'tv/monitor'])

true_labels = np.load('../data/true_labels.npy')
pred_labels = np.load('../data/pred_labels.npy')

true_labels = np.argmax(true_labels, axis=-1)
pred_labels = np.argmax(pred_labels, axis=-1)

true_classes = np.zeros(20)
pred_classes = np.zeros(20)

for i in range(1449):
    true_label = true_labels[i]
    pred_label = pred_labels[i]

    for n in range(5):
        true_l = true_label[n]
        pred_l = pred_label[n]

        if true_l == 20:
            continue

        true_classes[true_l] += 1
        if true_l == pred_l:
            pred_classes[pred_l] += 1

accuracy = pred_classes / true_classes * 100
index = list(reversed(np.argsort(accuracy)))
accuracy = accuracy[index]
print('Mean Accuracy: {0:.2f}%'.format(np.mean(accuracy)))

classes = classes[index]
colors = np.asarray(random_colors(20))
colors = colors[index]

plt.bar(classes, accuracy, color=colors)
plt.xticks(rotation=45)
plt.ylim([0, 80])
plt.ylabel('Accuracy (%)')
plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.98, wspace=0, hspace=0)
plt.savefig('../figures/accuracy.eps', bbox_inches="tight", pad_inches=0)
plt.show()
