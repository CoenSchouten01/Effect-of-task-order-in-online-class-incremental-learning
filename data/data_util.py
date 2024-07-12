from collections import defaultdict
import torch
from torch.utils.data import Subset
from torch.utils.data.dataset import TensorDataset


def take_subset(dataset, sample_size, num_classes):
    """
    Function to return a subset of the original dataset as TensorDataset.

    Args:
        - dataset: The original dataset of which to take the subset
        - sample_size: Desired number of samples per class
        - num_classes: Number of classes in the dataset

    Returns:
        - TensorDataset, subset of the original dataset
    """
    class_ids = list(range(num_classes))
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
    return TensorDataset(torch.cat(data), torch.cat(labels))

def get_cifar100_superclass_tasks():
    return [
        [4, 30, 55, 72, 95], # Aquactic Mammals
        [1, 32, 67, 73, 91], # fish
        [54, 62, 70, 82, 92], # flowers
        [9, 10, 16, 28, 61], # food containers
        [0, 51, 53, 57, 83], # fruit and vegetables
        [22, 39, 40, 86, 87], # household electrical devices
        [5, 20, 25, 84, 94], # household furniture
        [6, 7, 14, 18, 24], # insects
        [3, 42, 43, 88, 97], # large carnivores
        [12, 17, 37, 68, 76], # large man-made outdoor things
        [23, 33, 49, 60, 71], # large natural outdoor scenes
        [15, 19, 21, 31, 38], # large omnivores and herbivores
        [34, 63, 64, 66, 75], # medium-sized mammals
        [26, 45, 77, 79, 99], # non-insect invertebrates
        [2, 11, 35, 46, 98], # people
        [27, 29, 44, 78, 93], # reptiles
        [36, 50, 65, 74, 80], # small mammals
        [47, 52, 56, 59, 96], # trees
        [8, 13, 48, 58, 90], # vehicles 1
        [41, 69, 81, 85, 89]  # vehicles 2      
        ]