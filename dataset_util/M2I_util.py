from avalanche.benchmarks.datasets import MNIST, Omniglot, FashionMNIST, SVHN, CIFAR10, TinyImagenet
from torch.utils.data import Dataset, ConcatDataset, TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from collections import defaultdict
import torch

from data.transform_util import mnist_transform, fashion_mnist_transform, svhn_transform, cifar10_transform, tinyimagenet_transform

def subsample_dataset(dataset, class_ids, sample_size):
    class_indices = defaultdict(list)
    # Store the indices for each class
    for i, (_, target) in enumerate(dataset):
        class_indices[target].append(i)
        if all([len(class_indices[class_id]) >= sample_size for class_id in class_ids]):
            break

    data = []
    labels = []
    for class_id in class_ids:
        indices = class_indices[class_id]
        if sample_size <= len(indices):
            indices = torch.Tensor(indices)
            # select sample_size indices
            indices = indices[:sample_size]
        else:
            raise Exception("Sample size too large for class: ", class_id)
        class_data = [dataset[int(i)][0] for i in indices]
        data.append(torch.stack(class_data))
        class_labels = [dataset[int(i)][1] for i in indices]
        labels.append(torch.tensor(class_labels).long())
    # Create the dataset for the entire class
    return data, labels


def get_M2I_dataset(train=True, classes_per_task=10, samples_per_class=500, image_shape=(64, 64)):
    dataset_constructors = [MNIST, FashionMNIST, SVHN, CIFAR10, TinyImagenet]
    dataset_transforms = [mnist_transform(True, image_shape), fashion_mnist_transform(True, image_shape), svhn_transform(True, image_shape), cifar10_transform(True, image_shape), tinyimagenet_transform(True, image_shape)]
    data = []
    labels = []
    for i, dataset_constructor in enumerate(dataset_constructors):
        if dataset_constructor != SVHN:
            dataset = dataset_constructor("datasets", train=train, transform=dataset_transforms[i], target_transform=lambda x: x + i*classes_per_task, download=True)
        else: # SVHN does not have the train argument
            if train:
                dataset = dataset_constructor("datasets", split="train", transform=dataset_transforms[i], target_transform=lambda x: x + i*classes_per_task, download=True)
            else:
                dataset = dataset_constructor("datasets", split="test", transform=dataset_transforms[i], target_transform=lambda x: x + i*classes_per_task, download=True)
        class_ids = [x + i * classes_per_task for x in range(classes_per_task)]
        task_data, task_labels = subsample_dataset(dataset=dataset, class_ids=class_ids, sample_size=samples_per_class)
        data.append(torch.cat(task_data))
        labels.append(torch.cat(task_labels))
    return TensorDataset(torch.cat(data), torch.cat(labels))
        
