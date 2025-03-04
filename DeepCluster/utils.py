import torch
import numpy as np
from torch.utils.data import Sampler, data
from torchvision import transforms

class AverageMeter:
    """
    Keeps track of average values over time (e.g., loss, accuracy).
    Useful for monitoring training progress.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all tracking values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with new value `val`, weighted by `n`."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_features(dataloader, model, N, get_labels=False):
    """
    Computes feature representations for a dataset using a given model.
    
    Args:
        dataloader: DataLoader for input data.
        model: Neural network model.
        N: Total number of samples.
        get_labels: If True, returns labels along with features.
    
    Returns:
        Feature matrix (and optionally labels).
    """
    model.eval()
    labels = []

    for i, (input_tensor, label) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), requires_grad=False)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * 64: (i + 1) * 64] = aux
        else:
            features[i * 64:] = aux

        labels.append(label.numpy())

    labels = np.concatenate(labels)
    return (features, labels) if get_labels else features


class UnifLabelSampler(Sampler):
    """
    Custom sampler that ensures uniform label distribution across mini-batches.
    """
    def __init__(self, N, images_lists):
        self.N = N  # Total number of samples to draw.
        self.images_lists = images_lists  # List of images grouped by clusters.
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        """Generates a shuffled list of sample indices with uniform label distribution."""
        nmb_non_empty_clusters = sum(len(cluster) != 0 for cluster in self.images_lists)
        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for cluster in self.images_lists:
            if len(cluster) == 0:
                continue
            indexes = np.random.choice(cluster, size_per_pseudolabel, replace=(len(cluster) <= size_per_pseudolabel))
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = res.astype('int').tolist()
        return res[:self.N] if len(res) >= self.N else res + res[:self.N - len(res)]

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


def cluster_assign(images_lists, dataset):
    """
    Assigns pseudo-labels to dataset samples based on clustering results.
    
    Args:
        images_lists: List of lists containing clustered sample indices.
        dataset: Original dataset.
    
    Returns:
        Reassigned dataset with pseudo-labels.
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)


class ReassignedDataset(data.Dataset):
    """
    Custom dataset that reassigns pseudo-labels to samples based on clustering.
    """
    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        """Creates a list of tuples (image, pseudo-label)."""
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        return images

    def __getitem__(self, index):
        """Returns a transformed image and its pseudo-label."""
        img, pseudolabel = self.imgs[index]
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)






# class AverageMeter:
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

# def compute_features(dataloader, model, N, get_labels=False):
#     model.eval()
#     labels = []

#     for i, (input_tensor, label) in enumerate(dataloader):
#         input_var = torch.autograd.Variable(input_tensor.cuda(), requires_grad=False)
#         aux = model(input_var).data.cpu().numpy()

#         if i == 0:
#             features = np.zeros((N, aux.shape[1]), dtype='float32')

#         aux = aux.astype('float32')
#         if i < len(dataloader) - 1:
#             features[i * 64: (i + 1) * 64] = aux
#         else:
#             features[i * 64:] = aux

#         labels.append(label.numpy())

#     labels = np.concatenate(labels)
#     return (features, labels) if get_labels else features

# class UnifLabelSampler(Sampler):
#     def __init__(self, N, images_lists):
#         self.N = N
#         self.images_lists = images_lists
#         self.indexes = self.generate_indexes_epoch()

#     def generate_indexes_epoch(self):
#         nmb_non_empty_clusters = sum(len(cluster) != 0 for cluster in self.images_lists)
#         size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
#         res = np.array([])

#         for cluster in self.images_lists:
#             if len(cluster) == 0:
#                 continue
#             indexes = np.random.choice(cluster, size_per_pseudolabel, replace=(len(cluster) <= size_per_pseudolabel))
#             res = np.concatenate((res, indexes))

#         np.random.shuffle(res)
#         res = res.astype('int').tolist()
#         return res[:self.N] if len(res) >= self.N else res + res[:self.N - len(res)]

#     def __iter__(self):
#         return iter(self.indexes)

#     def __len__(self):
#         return len(self.indexes)

# def cluster_assign(images_lists, dataset):
#     assert images_lists is not None
#     pseudolabels = []
#     image_indexes = []
#     for cluster, images in enumerate(images_lists):
#         image_indexes.extend(images)
#         pseudolabels.extend([cluster] * len(images))

#     t = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.1307,), std=(0.3081,))
#     ])

#     return ReassignedDataset(image_indexes, pseudolabels, dataset, t)

# class ReassignedDataset(data.Dataset):
#     def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
#         self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
#         self.transform = transform

#     def make_dataset(self, image_indexes, pseudolabels, dataset):
#         label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
#         images = []
#         for j, idx in enumerate(image_indexes):
#             path = dataset[idx][0]
#             pseudolabel = label_to_idx[pseudolabels[j]]
#             images.append((path, pseudolabel))
#         return images

#     def __getitem__(self, index):
#         img, pseudolabel = self.imgs[index]
#         return img, pseudolabel

#     def __len__(self):
#         return len(self.imgs)
