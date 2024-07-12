from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from dataset_util.M2I_util import get_M2I_dataset
from data.transform_util import cifar10_transform

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
import torch

import torch.nn as nn
from utils.config import DEVICE
from copy import deepcopy
import numpy as np
import seaborn as sns
import colorcet as cc
import os

def subset_seen_classes(dataset, sample_size, seen_classes):
    class_indices = defaultdict(list)
    # Store the indices for each class
    for i, (_, target) in enumerate(dataset):
        class_indices[int(target)].append(i)
        if all([len(class_indices[class_id]) >= sample_size for class_id in seen_classes]):
            break

    data = []
    labels = []
    for class_id in seen_classes:
        indices = class_indices[class_id]
        indices = torch.Tensor(indices)
        if sample_size <= len(indices):
            # select sample_size indices
            indices = indices[:sample_size]
        class_data = [dataset[int(i)][0] for i in indices]
        data.append(torch.stack(class_data))
        class_labels = [dataset[int(i)][1] for i in indices]
        labels.append(torch.tensor(class_labels).long())
    # Create the dataset for the entire class
    return TensorDataset(torch.cat(data), torch.cat(labels))


def feature_tsne(model, dataset, dataset_name, strategy, tasks, task_order, position):
    """ 
    Function to plot and store TSNE embedding of the current model

    """
    model = deepcopy(model)
    # Prepare a subset of the dataset with the desired number of samples per class
    seen_tasks = [tasks[idx] for idx in task_order[:position]]
    seen_classes = [item for sublist in seen_tasks for item in sublist]
    subset = subset_seen_classes(dataset, sample_size = 500//len(seen_classes), seen_classes=seen_classes)

    # Define a DataLoader for the subset dataset
    batch_size = 64
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    feature_maps = []
    labels = []

    # Extract feature maps using the trained model
    # Visualize feature maps
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    if strategy != "cope":
        model.avgpool.register_forward_hook(get_activation('feature maps'))
    else:
        model.fc.register_forward_hook(get_activation('feature maps'))
    for batch_x, batch_y in subset_loader:
        _ = model(batch_x.to(DEVICE))
        feature_maps.append(torch.squeeze(activations['feature maps']).cpu())
        labels.append(batch_y)

    feature_maps = np.concatenate(feature_maps)
    labels = np.concatenate(labels)

    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(feature_maps)

    # Plot t-SNE embeddings colored per class
    plot_tsne(tsne_embeddings, labels, dataset_name, strategy, position, task_order, label_type="class_label")
    plot_tsne(tsne_embeddings, np.array(class_label_to_task_label(labels, tasks)), dataset_name, strategy, position, task_order, label_type="task_label")

def plot_tsne(tsne_embeddings, labels, dataset_name, strategy, position, task_order, label_type):
    if "CIFAR100" in dataset_name and label_type == "task_label":
        palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(labels)))
    elif len(set(labels)) < 10 and "CIFAR100" not in dataset_name:
        palette = {0: "tab:blue", 1: "tab:orange", 2:"tab:green", 3:"tab:red", 4:"tab:purple", 5:"tab:brown", 6:"tab:pink", 7:"tab:gray", 8:"tab:olive", 9:"tab:cyan"}
    else:
        palette = "deep"
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
            x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], hue=labels, palette=palette
        )
    plt.title(f't-SNE Plot of Feature Maps for {dataset_name} {strategy} {task_order[:position+1]}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    save_directory = f"figures/tsne/{dataset_name}/{strategy}/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_path = os.path.join(save_directory, f"feature_map_tsne_position_{position}_order_{task_order}_{label_type}.png")

    plt.savefig(save_path)

def class_label_to_task_label(class_labels, tasks):
    class_to_task = {}
    for i, task in enumerate(tasks):
        for class_label in task:
            class_to_task[class_label] = i

    # Transform the list of class labels to a list of task labels
    return [class_to_task[class_label] for class_label in class_labels] 