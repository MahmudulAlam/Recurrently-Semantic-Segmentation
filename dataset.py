import cv2
import torch
import random
import numpy as np
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

VOC_CLASSES = ['aeroplane',
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
               'tv/monitor',
               'background']

VOC_COLORMAP = [[128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
                [0, 0, 0]]


class PascalVOC(VOCSegmentation):
    def __init__(self, root='./data/', image_set='train', download=False, transform=None):
        super().__init__(root=root, image_set=image_set, download=download, transform=transform)

    def create_mask(self, mask, seed):
        height = 256
        width = 256
        n_classes = 21
        max_instances = 5
        new_mask = np.zeros((height, width, n_classes), dtype=np.float32)
        labels = np.ones((n_classes,), dtype=np.int32) * (n_classes - 1)

        index = 0
        for cls, label in enumerate(VOC_COLORMAP[:-1]):
            m = np.all(mask == label, axis=-1).astype(float)
            if np.sum(m) != 0:
                if self.transform is not None:
                    m = torch.tensor(m, dtype=torch.float32)
                    torch.manual_seed(seed)
                    m = self.transform(m)
                    m = np.asarray(m)
                    m = np.squeeze(m)
                    m = m / 255.
                    m = np.where(m >= 0.5, 1., 0.)
                new_mask[:, :, index] = m
                labels[index] = cls
                index += 1

        loss_mask = np.zeros((5,))
        loss_mask[0:index + 1] = 1.

        labels = np.eye(n_classes)[labels]
        new_mask = new_mask[:, :, 0:max_instances]
        labels = labels[0:max_instances, :]
        return new_mask, labels, loss_mask

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        seed = random.randint(0, 1000)
        if self.transform is not None:
            torch.manual_seed(seed)
            image = self.transform(image)
            image = np.asarray(image)

        mask = cv2.imread(self.masks[index])
        mask = cv2.resize(mask, (256, 256))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask, labels, loss_mask = self.create_mask(mask, seed)

        image = image.transpose((2, 0, 1)) / 255.
        mask = mask.transpose((2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, mask, labels, loss_mask


def load_dataset(root, batch_size, shuffle=True):
    pil = transforms.ToPILImage()
    crop = transforms.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.2))
    affine = transforms.RandomAffine(degrees=(-60, 60), translate=(0.0, .25), scale=(.7, 1.3))

    transform = transforms.Compose([pil, crop, affine])

    train_set = PascalVOC(root=root, image_set='train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                                               drop_last=False)

    test_set = PascalVOC(root=root, image_set='val')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=False)
    return train_loader, test_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def imshow(img):
        img = img.numpy()
        plt.figure()
        return img


    train, test = load_dataset(root='./data/', batch_size=32)

    for x, y, z, w in train:
        print(x.shape)
        print(y.shape)
        print(z.shape)
        print(w.shape)

        x = x[0].numpy()
        y = y[0].numpy()
        z = z[0].numpy()
        w = w[0].numpy()
        print(np.max(x), np.min(x))
        print(np.max(y), np.min(y))

        x = x.transpose((1, 2, 0))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.asarray(x * 255., dtype=np.uint8)

        plt.imsave('./outputs/image.jpg', x)
        print(w)
        for i in range(5):
            plt.imshow(y[i])
            plt.show()
            plt.imsave('./outputs/mask_' + str(i) + '.jpg', y[i])
            print(z[i])
        break
    plt.show()
