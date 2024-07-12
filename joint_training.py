from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10, CIFAR100
from dataset_util.M2I_util import get_M2I_dataset
from data.transform_util import cifar10_transform, cifar100_transform
import colorcet as cc
from scipy.spatial.distance import cosine
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
import torch
from data.data_util import get_cifar100_superclass_tasks

import torch.nn as nn
from utils.config import DEVICE
from itertools import permutations

import numpy as np
import seaborn as sns


def get_per_class_accuracy(dataset_name):
    if dataset_name == "M2I":
        train_set = get_M2I_dataset(train=True)
        test_set = get_M2I_dataset(train=False, samples_per_class=50)
        num_classes = 50
    elif dataset_name in ["CIFAR10", "CIFAR10-random"]:
        train_set = CIFAR10(
            "datasets",
            train=True,
            transform=cifar10_transform(resize=False),
            download=True,
        )
        test_set = CIFAR10(
            "datasets",
            train=False,
            transform=cifar10_transform(resize=False),
            download=True,
        )
        num_classes = 10
    elif dataset_name in ["CIFAR100", "CIFAR100-superclass"]:
        train_set = CIFAR100(
            "datasets",
            train=True,
            transform=cifar100_transform(resize=False),
            download=True,
        )
        test_set = CIFAR100(
            "datasets",
            train=False,
            transform=cifar100_transform(resize=False),
            download=True,
        )
        num_classes = 100

    model = fit(train_set, num_classes)

    test_dl = DataLoader(test_set, batch_size=32, shuffle=True)

    per_class_accuracy = eval(model, test_dl, num_classes=num_classes)
    return per_class_accuracy


def feature_visualization(dataset_name, samples_per_class=50, random_seed=42):
    np.random.seed(random_seed)
    if dataset_name == "M2I":
        train_set = get_M2I_dataset(train=True)
        num_classes = 50
    elif dataset_name == "CIFAR100":
        train_set = CIFAR100(
            "datasets",
            train=True,
            transform=cifar100_transform(resize=True),
            download=True,
        )
        num_classes = 100
    elif "CIFAR10" in dataset_name:
        train_set = CIFAR10(
            "datasets",
            train=True,
            transform=cifar10_transform(resize=False),
            download=True,
        )
        num_classes = 10

    model = resnet18(weights="IMAGENET1K_V1").to(DEVICE)
    model.eval()

    # Prepare a subset of the dataset with the desired number of samples per class
    subset_indices = []
    for label in range(num_classes):
        if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
            label_indices = np.where(np.array(train_set.targets) == label)[0]
        elif dataset_name == "M2I":
            label_indices = np.where(np.array(train_set.tensors[1]) == label)[0]
        subset_indices.extend(
            np.random.choice(label_indices, samples_per_class, replace=False)
        )

    subset_dataset = torch.utils.data.Subset(train_set, subset_indices)

    # Define a DataLoader for the subset dataset
    batch_size = 64
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    feature_maps = []
    labels = []

    # Extract feature maps using the trained model
    # Visualize feature maps
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    model.avgpool.register_forward_hook(get_activation("feature maps"))
    for batch_x, batch_y in subset_loader:
        _ = model(batch_x.to(DEVICE))
        feature_maps.append(torch.squeeze(activations["feature maps"]).cpu())
        labels.append(batch_y)

    feature_maps = np.concatenate(feature_maps)
    labels = np.concatenate(labels)

    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(feature_maps)

    # Plot t-SNE embeddings colored per task
    plot_tsne(tsne_embeddings, labels, dataset_name, label_type="class_label")

    # Plot t-SNE embeddings colored per task
    if dataset_name == "CIFAR10":
        # regular split
        tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        print("CIFAR10")
        task_labels = np.array(class_label_to_task_label(labels, tasks))
        calculate_task_embedding_characteristics(feature_maps, task_labels)
        plot_tsne(tsne_embeddings, task_labels, dataset_name, label_type="task_label")

        # random split
        tasks = [[7, 3], [2, 8], [5, 6], [9, 4], [0, 1]]
        print("CIFAR10-random")
        task_labels = np.array(class_label_to_task_label(labels, tasks))
        calculate_task_embedding_characteristics(feature_maps, task_labels)
        plot_tsne(
            tsne_embeddings,
            task_labels,
            dataset_name + "-random",
            label_type="task_label",
        )
    if dataset_name == "CIFAR100":
        classes = list(range(num_classes))
        classes_per_task = 5
        num_tasks = num_classes // classes_per_task
        tasks = [
            classes[i * classes_per_task : (i + 1) * classes_per_task]
            for i in range(num_tasks)
        ]
        task_labels = np.array(class_label_to_task_label(labels, tasks))
        calculate_task_embedding_characteristics(feature_maps, task_labels)
        plot_tsne(tsne_embeddings, task_labels, dataset_name, label_type="task_label")
        tasks = get_cifar100_superclass_tasks()
        task_labels = np.array(class_label_to_task_label(labels, tasks))
        plot_tsne(
            tsne_embeddings,
            task_labels,
            dataset_name + "-superclass",
            label_type="task_label",
        )
    if dataset_name == "M2I":
        classes = list(range(num_classes))
        classes_per_task = 10
        num_tasks = num_classes // classes_per_task
        tasks = [
            classes[i * classes_per_task : (i + 1) * classes_per_task]
            for i in range(num_tasks)
        ]
        task_labels = np.array(class_label_to_task_label(labels, tasks))
        plot_tsne(tsne_embeddings, task_labels, dataset_name, label_type="task_label")


def calculate_task_embedding_characteristics(feature_maps, labels):
    feature_maps_per_label = defaultdict(list)
    for feature_map, label in zip(feature_maps, labels):
        feature_maps_per_label[label].append(feature_map)

    mean_embeddings = dict()
    for label, embeddings in feature_maps_per_label.items():
        embeddings = np.array(embeddings)
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embeddings[label] = mean_embedding


def class_label_to_task_label(class_labels, tasks):
    class_to_task = {}
    for i, task in enumerate(tasks):
        for class_label in task:
            class_to_task[class_label] = i

    # Transform the list of class labels to a list of task labels
    return [class_to_task[class_label] for class_label in class_labels]

def get_unique_ordered(entries):
    seen = set()
    unique_entries = []
    
    for entry in entries:
        if entry not in seen:
            seen.add(entry)
            unique_entries.append(entry)
    
    return unique_entries

def plot_tsne(tsne_embeddings, labels, dataset_name, label_type):
    plt.figure(figsize=(10, 10))
    labels = [f"Task {label}" for label in labels]
    if "CIFAR100" in dataset_name:
        palette = sns.color_palette(cc.glasbey, n_colors=20)
    else:
        palette = "deep"

    legend_labels = [f'Task {label}' for label in labels]
    sns.scatterplot(
        x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], hue=labels, palette=palette
    )
    dataset_to_name = {
        "CIFAR10": "CIFAR10-I",
        "CIFAR10-random": "CIFAR10-II",
        "M2I": "M2I",
    }
    plt.title(f"t-SNE Plot of Feature Maps for {dataset_to_name[dataset_name]}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.savefig(f"results/analysis/characteristics_good_bad_orders/{dataset_name}_feature_maps_{label_type}.png")


def fit(train_set, num_classes, num_epochs=1):
    model = resnet18()
    model.fc = nn.Linear(512, num_classes)
    model.to(DEVICE)

    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = CrossEntropyLoss()

    train_dl = DataLoader(train_set, batch_size=32, shuffle=True)
    model.train()

    for _ in range(num_epochs):
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


def eval(model, test_dl, num_classes):
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(predicted)):
                label = labels[i]
                prediction = predicted[i]
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1
    per_class_accuracy = [class_correct[i] / class_total[i] for i in range(num_classes)]
    return per_class_accuracy


def get_task_dataset(dataset, task):
    class_indices = {}
    # Store the indices for each class
    for i, (_, target) in enumerate(dataset):
        target = int(target)
        if target not in class_indices:
            class_indices[target] = []
        class_indices[target].append(i)

    task_data = []
    task_labels = []
    # select all samples from each class in that task
    for idx, class_id in enumerate(task):
        indices = class_indices[class_id]
        class_data = [dataset[int(i)][0] for i in indices]
        task_data.append(torch.stack(class_data))
        class_labels = [idx for i in indices]
        task_labels.append(torch.tensor(class_labels).long())
    # Create the dataset for the entire class
    return TensorDataset(torch.cat(task_data), torch.cat(task_labels))


def train_one_task(dataset_name, task):
    if dataset_name == "CIFAR10":
        train_set = CIFAR10(
            "datasets",
            train=True,
            transform=cifar10_transform(resize=False),
            download=True,
        )
        test_set = CIFAR10(
            "datasets",
            train=False,
            transform=cifar10_transform(resize=False),
            download=True,
        )
    elif dataset_name == "CIFAR100":
        train_set = CIFAR100(
            "datasets",
            train=True,
            transform=cifar100_transform(resize=False),
            download=True,
        )
        test_set = CIFAR100(
            "datasets",
            train=False,
            transform=cifar100_transform(resize=False),
            download=True,
        )
    elif dataset_name == "M2I":
        train_set = get_M2I_dataset(train=True)
        test_set = get_M2I_dataset(train=False, samples_per_class=50)
    else:
        raise Exception("Incorrect dataset name")

    task_train_set = get_task_dataset(train_set, task)
    task_test_set = get_task_dataset(test_set, task)

    task_test_dl = DataLoader(task_test_set, batch_size=32, shuffle=True)

    model = fit(task_train_set, len(task))
    return eval(model, task_test_dl, len(task))


if __name__ == "__main__":
    datasets = ["M2I", "CIFAR10", "CIFAR10-random"]
    
    feature_visualization("CIFAR10", samples_per_class=100)
    feature_visualization("M2I", samples_per_class=50)
    

    for dataset in datasets:
        if dataset == "CIFAR10-random":
            dataset_name = "CIFAR10"
            tasks = [[7, 3], [2,8], [5,6], [9, 4], [0, 1]]
        elif dataset == "CIFAR10":
            dataset_name = "CIFAR10"
            tasks = [[0, 1], [2,3], [4,5], [6, 7], [8, 9]]
        elif dataset == "M2I":
            dataset_name = "M2I"
            num_classes = 50
            classes = list(range(num_classes))
            classes_per_task = 10
            num_tasks = num_classes // classes_per_task
            tasks = [
                        classes[i * classes_per_task : (i + 1) * classes_per_task]
                        for i in range(num_tasks)
                    ]
        per_task_avg_accuracy = []
        for task in tasks:
            task_per_class_accuracy = train_one_task(dataset_name, task)
            print(f"accuracies of task {task} ", task_per_class_accuracy)
            per_task_avg_accuracy.append(np.mean(task_per_class_accuracy))
        print(f"{dataset} per task avg accuracy: ", per_task_avg_accuracy)

    for dataset in datasets:
        if dataset == "CIFAR10-random":
            dataset_id = 0
            tasks = [[7, 3], [2,8], [5,6], [9, 4], [0, 1]]
        elif dataset == "CIFAR10":
            dataset_id = 0
            tasks = [[0, 1], [2,3], [4,5], [6, 7], [8, 9]]
        elif dataset == "M2I":
            dataset_id = 1
            num_classes = 50
            classes = list(range(num_classes))
            classes_per_task = 10
            num_tasks = num_classes // classes_per_task
            tasks = [
                        classes[i * classes_per_task : (i + 1) * classes_per_task]
                        for i in range(num_tasks)
                    ]
        elif dataset == "CIFAR100":
            num_classes = 100
            classes = list(range(num_classes))
            classes_per_task = 5
            num_tasks = num_classes // classes_per_task
            tasks = [
                        classes[i * classes_per_task : (i + 1) * classes_per_task]
                        for i in range(num_tasks)
                    ]
        elif dataset == "CIFAR100-superclass":
            tasks = get_cifar100_superclass_tasks()
        per_task_avg_accuracy = []
        per_class_accuracies = get_per_class_accuracy(dataset_name=dataset)
        for task in tasks:
            task_accuracy = []
            for class_id in task:
                task_accuracy.append(per_class_accuracies[class_id])
            per_task_avg_accuracy.append(np.mean(task_accuracy))
        print(f"{dataset} joint-training per class avg accuracy: ", per_class_accuracies)
        print(f"{dataset} joint-training per task avg accuracy: ", per_task_avg_accuracy)
