import cv2
import time
import torch
import numpy as np
from network import UNet
from visualize import random_colors, apply_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

network = UNet(n_channels=3, n_classes=1)
network.eval()
network.to(device)
network.load_state_dict(torch.load('./weights/weights.h5'))


def preprocess(x):
    x = cv2.resize(x, (256, 256))
    x = x.transpose((2, 0, 1)) / 255.
    x = torch.tensor(x, dtype=torch.float32)
    x = torch.unsqueeze(x, dim=0)
    x = x.to(device)
    return x


def predict(img):
    with torch.no_grad():
        img = preprocess(img)
        masks, labels = network(img)
        masks = masks[0].detach().cpu().numpy()
        labels = labels[0].detach().cpu().numpy()
    return masks, labels


cam = cv2.VideoCapture(0)
colors = random_colors(21)

while True:
    ret, frame = cam.read()

    if ret is False:
        break

    tic = time.time()
    image = cv2.flip(frame, 1)
    height, width, _ = image.shape

    masks, labels = predict(image)
    labels = labels.argmax(-1)

    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask = cv2.resize(mask, (width, height))
        mask = np.where(mask >= 0.3, 1., 0.)
        if mask.sum() == 0:
            break
        image = apply_mask(image, mask, colors[label])
        if label == 20:
            break

    cv2.imshow('webcam', image)
    toc = time.time()
    print(labels, '{0:.2f} FPS'.format(1. / (toc - tic)))
    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
