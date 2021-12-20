import cv2
import torch
import numpy as np
from network import UNet
import matplotlib.pyplot as plt
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
    x = torch.cat([x for _ in range(8)], dim=0)
    x = x.to(device)
    return x


filename = '2010_001131.jpg'
image = cv2.imread('./data/images/' + filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape

with torch.no_grad():
    img = preprocess(image)
    masks, labels = network(img)
    masks = masks[0].detach().cpu().numpy()
    labels = labels[0].detach().cpu().numpy()

threshold = 0.5
colors = random_colors(21)
labels = labels.argmax(-1)
print(labels)

for i, (mask, label) in enumerate(zip(masks, labels)):
    mask = cv2.resize(mask, (width, height))
    mask = np.where(mask >= 0.5, 1., 0.)
    image = apply_mask(image, mask, colors[label])
    plt.imsave('./outputs/mask_' + str(i) + '.jpg', mask)

plt.imsave('./outputs/' + filename, image)
