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
true_masks = np.load('../data/true_masks.npy')
pred_masks = np.load('../data/pred_masks.npy')

true_labels = np.argmax(true_labels, axis=-1)
pred_labels = np.argmax(pred_labels, axis=-1)
true_masks = np.where(true_masks >= 0.5, 1., 0.)
pred_masks = np.where(pred_masks >= 0.5, 1., 0.)

iou_classes = np.zeros(20)
pred_classes = np.zeros(20)

for i in range(1449):
    true_label = true_labels[i]
    pred_label = pred_labels[i]
    true_mask = true_masks[i]
    pred_mask = pred_masks[i]

    for n in range(5):
        true_l = true_label[n]
        pred_l = pred_label[n]
        true_m = true_mask[n]
        pred_m = pred_mask[n]

        if true_l == 20:
            continue

        if true_l == pred_l:
            intersection = np.sum(np.logical_and(true_m, pred_m))
            union = np.sum(np.logical_or(true_m, pred_m))
            iou = (intersection / (union + 1e-10))
            iou_classes[pred_l] += iou
            pred_classes[pred_l] += 1

mean_iou = iou_classes / pred_classes * 100
index = list(reversed(np.argsort(mean_iou)))
mean_iou = mean_iou[index]
print('Mean IOU: {0:.2f}%'.format(np.mean(mean_iou)))

classes = classes[index]
colors = np.asarray(random_colors(20))
colors = colors[index]

plt.bar(classes, mean_iou, color=colors)
plt.xticks(rotation=45)
plt.ylim([0, 80])
plt.ylabel('Mean IOU (%)')
plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.98, wspace=0, hspace=0)
plt.savefig('../figures/mean_iou.eps', bbox_inches="tight", pad_inches=0)
plt.show()
