import colorsys
import numpy as np


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for i in range(3):
        image[:, :, i] = np.where(mask == 1, image[:, :, i] * (1 - alpha) + alpha * color[i] * 255, image[:, :, i])
    return image


def apply_contour(image, mask, color, thickness=4):
    t = thickness // 2
    mask = mask.copy()
    mask1 = mask[thickness:, thickness:]
    mask2 = mask[:-thickness, :-thickness]
    mask[t:-t, t:-t] -= mask1 * mask2
    mask = np.where(mask == 0, 0., 1.)
    image = apply_mask(image, mask, color, alpha=1.)
    return image
