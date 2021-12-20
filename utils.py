import torch


def cross_entropy_loss(y_true, y_pred, mask=None, eps=1e-10):
    y_t = torch.clip(y_true, eps, 1.)
    y_p = torch.clip(y_pred, eps, 1.)

    y_t_prime = torch.clip(1. - y_true, eps, 1.)
    y_p_prime = torch.clip(1. - y_pred, eps, 1.)

    loss = - torch.xlogy(y_t, y_p) - torch.xlogy(y_t_prime, y_p_prime)
    loss = torch.mean(loss, dim=[-2, -1])
    loss = loss * mask
    loss = torch.sum(loss) / torch.sum(mask)
    return loss


def classifier_loss(y_true, y_pred, eps=1e-10):
    y_t = torch.clip(y_true, eps, 1.)
    y_p = torch.clip(y_pred, eps, 1.)
    loss = - y_t * torch.log(y_p)
    loss = torch.mean(loss)
    return loss


def dice_loss(y_true, y_pred, smooth=1, mask=None):
    intersection = torch.sum(y_true * y_pred, dim=[2, 3])
    union = torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1. - dice
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def evaluate(y_true, y_pred):
    y_true = torch.argmax(y_true, dim=-1)
    y_pred = torch.argmax(y_pred, dim=-1)
    accuracy = torch.mean(torch.all(y_true == y_pred, dim=-1).float())
    return accuracy


def create_indices(batch_size, dataset_size):
    index_i = list(range(0, dataset_size, batch_size))
    index_j = list(range(batch_size, dataset_size, batch_size))
    index_j.append(dataset_size)
    indices = list(zip(index_i, index_j))
    return indices
