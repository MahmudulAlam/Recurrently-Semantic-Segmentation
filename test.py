import os
import time
import torch
import numpy as np
from network import UNet
from dataset import load_dataset
from utils import create_indices

if os.name == 'nt':
    root = './data/'
else:
    root = './../../oates_common/mohammad/pascal_voc/'

train_loader, test_loader = load_dataset(root=root, batch_size=32)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

network = UNet(n_channels=3, n_classes=1)
network.to(device)
network.load_state_dict(torch.load('./weights/weights.h5'))

total, correct = 0, 0
mean_acc, mean_iou = 0, 0
tic = time.time()

stack_images = np.zeros((1449, 3, 256, 256))
stack_true_masks = np.zeros((1449, 5, 256, 256))
stack_pred_masks = np.zeros((1449, 5, 256, 256))
stack_true_labels = np.zeros((1449, 5, 21))
stack_pred_labels = np.zeros((1449, 5, 21))
indices = create_indices(40, 1449)

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        a, b = indices[i]
        images, true_masks, true_labels = data[0].to(device), data[1].to(device), data[2].to(device)
        loss_masks = data[3].to(device)

        # forward + loss + backward + optimize
        pred_masks, pred_labels = network(images)

        stack_images[a:b] = images.detach().cpu().numpy()
        stack_true_masks[a:b] = true_masks.detach().cpu().numpy()
        stack_pred_masks[a:b] = pred_masks.detach().cpu().numpy()
        stack_true_labels[a:b] = true_labels.detach().cpu().numpy()
        stack_pred_labels[a:b] = pred_labels.detach().cpu().numpy()

        pred_masks = torch.where(pred_masks >= 0.5, 1., 0.)
        true_labels = torch.argmax(true_labels, dim=-1)
        pred_labels = torch.argmax(pred_labels, dim=-1)

        acc_mask = (true_labels == pred_labels).float()
        correct += torch.sum(torch.all(acc_mask, dim=-1).float()).item()
        total += acc_mask.shape[0]

        intersection = torch.sum(torch.logical_and(true_masks, pred_masks), dim=[-2, -1])
        union = torch.sum(torch.logical_or(true_masks, pred_masks), dim=[-2, -1])
        iou = (intersection / (union + 1e-10)) * loss_masks
        iou = torch.sum(iou, dim=-1) / torch.sum(loss_masks, dim=-1)
        mean_iou += torch.sum(iou).item()

    mean_acc = (correct / total) * 100
    mean_iou = (mean_iou / total) * 100
    toc = time.time()
    form = 'Accuracy: {0:>.2f}%  Mean IOU: {1:>.2f}%  ETC: {2:>.2f}s'
    print(form.format(mean_acc, mean_iou, toc - tic))

np.save('data/images.npy', stack_images)
np.save('data/true_masks.npy', stack_true_masks)
np.save('data/pred_masks.npy', stack_pred_masks)
np.save('data/true_labels.npy', stack_true_labels)
np.save('data/pred_labels.npy', stack_pred_labels)
