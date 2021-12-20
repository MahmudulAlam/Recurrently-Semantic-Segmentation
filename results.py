import numpy as np
import matplotlib.pyplot as plt
from visualize import random_colors, apply_mask

images = np.load('./data/images.npy')
masks = np.load('./data/pred_masks.npy')
labels = np.load('./data/pred_labels.npy')
colors = random_colors(21)

for i in range(1449):
    image = images[i]
    mask = masks[i]
    label = labels[i]

    image = np.transpose(image, (1, 2, 0))
    image = np.asarray(image * 255., dtype=np.uint8)
    mask = np.where(mask >= 0.5, 1., 0.)
    label = np.argmax(label, axis=-1)
    print(label)
    for n in range(5):
        if label[n] == 20:
            break
        image = apply_mask(image, mask[n], colors[label[n]])

    plt.imsave('outputs/image_' + str(i) + '.jpg', image)
