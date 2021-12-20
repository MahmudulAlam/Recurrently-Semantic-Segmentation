import os
import time
import torch
from network import UNet
from dataset import load_dataset
from utils import cross_entropy_loss, dice_loss, classifier_loss

if os.name == 'nt':
    root = './data/'
else:
    root = './../../oates_common/mohammad/pascal_voc/'

train_loader, test_loader = load_dataset(root=root, batch_size=32)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

network = UNet(n_channels=3, n_classes=1)
network.to(device)
network.load_state_dict(torch.load('weights/weights.h5'))

epochs = 1000
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    loss, total, correct = 0, 0, 0
    mean_acc, mean_iou = 0, 0
    tic = time.time()

    for i, data in enumerate(test_loader, 0):
        images, true_masks, true_labels = data[0].to(device), data[1].to(device), data[2].to(device)
        loss_masks = data[3].to(device)
        optimizer.zero_grad()

        # forward + loss + backward + optimize
        pred_masks, pred_labels = network(images)
        loss = cross_entropy_loss(true_masks, pred_masks, mask=loss_masks)
        loss += classifier_loss(true_labels, pred_labels)
        loss += dice_loss(true_masks, pred_masks, mask=loss_masks)
        loss.backward()
        optimizer.step()

        pred_masks = torch.where(pred_masks >= 0.5, 1., 0.)
        true_labels = torch.argmax(true_labels, dim=-1)
        pred_labels = torch.argmax(pred_labels, dim=-1)

        acc_mask = (true_labels == pred_labels).float()
        correct += torch.sum(torch.all(acc_mask, dim=-1).float()).item()
        total += acc_mask.shape[0]

        intersection = torch.sum(torch.logical_and(true_masks, pred_masks), dim=[-2, -1]) * acc_mask
        union = torch.sum(torch.logical_or(true_masks, pred_masks), dim=[-2, -1])
        iou = (intersection / (union + 1e-10)) * loss_masks
        iou = torch.sum(iou, dim=-1) / torch.sum(loss_masks, dim=-1)
        mean_iou += torch.sum(iou).item()

    mean_acc = (correct / total) * 100
    mean_iou = (mean_iou / total) * 100
    toc = time.time()
    form = 'Epoch: {0:>3d}/{1}  Train Loss: {2:>8.6f}  Accuracy: {3:>.2f}%  Mean IOU: {4:>.2f}%  ETC: {5:>.2f}s'
    print(form.format(epoch, epochs, loss, mean_acc, mean_iou, toc - tic))

    if epoch % 10 == 0:
        torch.save(network.state_dict(), './weights/weights.h5')
