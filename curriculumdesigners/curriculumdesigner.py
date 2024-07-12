from abc import abstractmethod
import torch
import numpy as np
import math
from scipy.spatial.distance import cosine, euclidean
from torchvision.models import resnet18, squeezenet1_1
from utils.config import DEVICE
from torch.utils.data import DataLoader, TensorDataset

from itertools import permutations
from random import randint

def create_task_datasets(dataset, tasks, sample_size):
    class_indices = {}
    # Store the indices for each class
    for i, (_, target) in enumerate(dataset):
        target = int(target)
        if target not in class_indices:
            class_indices[target] = []
        class_indices[target].append(i)

    task_datasets = []
    # for every task
    for task in tasks:
        task_data = []
        task_labels = []
        # select sample_size samples from each class in that task
        for class_id in task:
            indices = class_indices[class_id]
            if sample_size < len(indices):
                indices = torch.Tensor(indices)
                # select sample_size indices
                indices = indices[:sample_size]
            class_data = [dataset[int(i)][0] for i in indices]
            task_data.append(torch.stack(class_data))
            class_labels = [dataset[int(i)][1] for i in indices]
            task_labels.append(torch.tensor(class_labels).long())
        # Create the dataset for the entire class
        task_dataset = TensorDataset(torch.cat(task_data), torch.cat(task_labels))
        task_datasets.append(task_dataset)
    return task_datasets


def compute_feature_embeddings(task_datasets, embedder):
    # compute the embedding for all tasks
    avg_vectors = []
    for task_dataset in task_datasets:
        task_vectors = []
        for batch in task_dataset:
            task_vectors.append(embedder(batch[0].to(DEVICE)))
        avg_vectors.append(
            torch.mean(torch.cat(task_vectors), axis=0).detach().cpu().numpy()
        )
    return avg_vectors

def compute_feature_deviations(task_datasets, embedder):
    # compute standard deviation of all task embeddings
    stds = []
    for task_dataset in task_datasets:
        task_vectors = []
        for batch in task_dataset:
            task_vectors.append(embedder(batch[0].to(DEVICE)))
        stds.append(torch.std(torch.cat(task_vectors)).detach().cpu().numpy())
    return stds

class CurriculumDesigner:
    """
    The Abstract base class for the curriculum designers
    """

    def __init__(self):
        pass

    @abstractmethod
    def generate_next_task(self, available_tasks):
        ...

class FixedCurriculumDesigner(CurriculumDesigner):
    def __init__(self, order):
        self.task_ordering = order
        self.task_index = 0
    
    def generate_next_task(self, available_tasks):
        next_task = self.task_ordering[self.task_index]
        available_tasks.remove(next_task)
        self.task_index += 1
        return next_task
    
    def get_ordering(self):
        return self.task_ordering

class RandomCurriculumDesigner(CurriculumDesigner):
    """
    This class is used as a baseline curriculum designer,
    it has one function, generate_next_task, which randomly
    selects the next task to train on.
    """

    def __init__(self):
        pass

    def generate_next_task(self, available_tasks):
        """
        Function to randomly generate the next task id from the set of available tasks
        This function also removes the selected task from the list of available tasks

        Args:
            available_tasks: List of unseen task id's

        Returns:
            The id of the next task to train on
        """
        task_id = np.random.choice(available_tasks)
        available_tasks.remove(task_id)
        return task_id

class L2LCurriculumDesigner(CurriculumDesigner):
    """
    The L2L Curriculum designer by Singh et al.
    This curriculum designer scores all possible curricula according to a score function
    and returns the curriculum with the highest score
    """
    def __init__(self, dataset, tasks, sample_size=500):
        # define the embedder
        self.embedder = resnet18(weights="IMAGENET1K_V1").to(DEVICE).eval()

        # Replace fully connected classification layer by identity
        self.embedder.fc = torch.nn.Identity()

        # freeze the embedder
        for param in self.embedder.parameters():
            param.requires_grad = False

        # create subdatasets for all tasks
        task_datasets = create_task_datasets(
            dataset=dataset, tasks=tasks, sample_size=sample_size
        )
        # create dataloader for each dataset
        for i, task_dataset in enumerate(task_datasets):
            task_datasets[i] = DataLoader(task_dataset, batch_size=32, shuffle=False)

        # compute the embedding for all tasks
        avg_vectors = compute_feature_embeddings(task_datasets, self.embedder)

        # compute the distances
        dataset_distances = [[] for _ in range(len(tasks))]
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                dist = cosine(avg_vectors[i], avg_vectors[j])
                dataset_distances[i].append(dist)

        # compute the variance in distances to all other tasks
        distance_var_list = []
        for i in range(len(tasks)):
            distances_to_other_tasks = []
            for j in range(len(tasks)):
                if i != j:
                    distances_to_other_tasks.append(dataset_distances[i][j])
            distance_var_list.append(np.var(distances_to_other_tasks))

        # compute and rank all permutations
        possible_orderings = list(permutations(range(len(tasks))))
        ordering_scores = [
            self.score_ordering(ordering, dataset_distances, distance_var_list)
            for ordering in possible_orderings
        ]
        self.task_ordering = possible_orderings[np.argmax(ordering_scores)]
        self.task_index = 0

    def generate_next_task(self, available_tasks):
        """
        Function to generate the next task id from the set of available tasks, based on precomputed greedy ordering
        This function also removes the selected task from the list of available tasks

        Args:
            available_tasks: List of unseen task id's

        Returns:
            The id of the next task to train on
        """
        next_task = self.task_ordering[self.task_index]
        available_tasks.remove(next_task)
        self.task_index += 1
        return next_task

    def score_ordering(self, ordering, dataset_distances, distance_var_list):
        score = 0
        # at t = 0
        score = 1 - distance_var_list[ordering[0]]
        for t in range(1, len(ordering)):
            if t <= math.floor(len(ordering) / 2):
                # distance to next task should be as large as possible
                score += dataset_distances[ordering[t]][ordering[t - 1]]
            else:
                # distance to task on opposite side of ordering as small as possbile
                score += (
                    1 - dataset_distances[ordering[t]][ordering[len(ordering) - t - 1]]
                )
        return score

    def get_ordering(self):
        return self.task_ordering
    
    
class OnTheFlyCurriculumDesigner(CurriculumDesigner):
    """ 
    This class implements the different variations of On-the-fly Curriculum Designers
    These curriculum designers take the current state of the continual learner into account for selecting
    the next task during training. There are different variations of this curriculum designer available.
    This curriculum designer does not have a get_ordering function, as it cannot compute the entire curriculum upfront.

    The following choices are available for this curriculum designer:
    1. First task selection: Easy / Center / Random
    2. Next task selection: Closest / Furthest
    3. Knowledge embedding: Mean of last task / Mean of all previous tasks

    The initial task gets selected based on feature embeddings of a pretrained ResNet18 model
    To select the next tasks the embedding of the current continual learner are used

    Arguments:
        dataset_name: the name of the dataset
        embedding_dataset: Dataset of images with size 224*224, based on which the first task will be selected
        train_set: the train set for the current CL setting, a subsample of this train set is used to determine the next task
        tasks: the composition of classes per task
        sample_size: number of samples per class to compute embeddings
        first_task: method of determining the first task, one of: "random", "easy" or "center"
        next_task: method of selecting the next task, either "min" or "max"
        average_embedding: when True, select next task based on average embedding of all previous tasks, when False, select only based on last task
    """

    def __init__(self,
        dataset_name,
        embedding_dataset,
        train_set,
        tasks,
        sample_size=50,
        first_task="random",
        next_task="max",
        average_embedding=False,
        ):
        # The method for determining the next task are only relevant in generate_next_task function
        self.next_task = next_task
        self.average_embedding = average_embedding 
        
        if first_task == "easy":
            # The easiest task is the one with the lowest norm in the task2vec embedding
            embedding_path = (
                f"embeddings/task2vec/{dataset_name}-{len(tasks)}-{len(tasks[0])}.pkl"
            )
            task2vec_embeddings = compute_task2vec_embeddings(embedding_path, embedding_dataset, tasks, sample_size)
            task_embedding_norms = [sum(task_embedding.hessian / task_embedding.scale) for task_embedding in task2vec_embeddings]
            self.initial_task = np.argmin(task_embedding_norms)
        elif first_task == "center":
            # The embedder for selecting the first task
            self.embedder = resnet18(weights="IMAGENET1K_V1").to(DEVICE).eval()

            # Replace fully connected classification layer by identity
            self.embedder.fc = torch.nn.Identity()

            # freeze the embedder
            for param in self.embedder.parameters():
                param.requires_grad = False

            # The task datasets used to determine the first task
            initial_embedding_task_datasets = create_task_datasets(dataset=embedding_dataset, tasks=tasks, sample_size=sample_size)
            
            # Create dataloaders
            for i, task_dataset in enumerate(initial_embedding_task_datasets):
                initial_embedding_task_datasets[i] = DataLoader(
                    task_dataset, batch_size=32, shuffle=False
                )
            task_embeddings = compute_feature_embeddings(initial_embedding_task_datasets, self.embedder)

            task_distances = [[] for _ in range(len(tasks))]

            # compute the distances
            for i in range(len(tasks)):
                for j in range(len(tasks)):
                    dist = cosine(task_embeddings[i], task_embeddings[j])
                    task_distances[i].append(dist)
            self.initial_task = np.argmin(
                [np.var(task_distance) for task_distance in task_distances]
            )
        elif first_task == "random":
            # The first task to train on is a random task
            self.initial_task = randint(0, len(tasks)-1)
        else:
            raise Exception("First task should be either random, easy or center")
        
        # Create a dataloader for each of the datasets per task used to determine the next task
        self.cl_embedding_task_datasets = create_task_datasets(dataset=train_set, tasks=tasks, sample_size=sample_size)
        for i, task_dataset in enumerate(self.cl_embedding_task_datasets):
            self.cl_embedding_task_datasets[i] = DataLoader(
                task_dataset, batch_size=32, shuffle=False
            )
        
        # variable to store the previously seen tasks
        self.task_history = []

    def generate_next_task(self, available_tasks, continual_learner):
        """
        Function to generate the next task id from the set of available tasks,
        based on the arguments passed to the constructor of this class.

        Args:
            available_tasks: List of unseen task id's
            continual_learner: The current state of the backbone model

        Returns:
            The id of the next task to train on
        """
        if len(self.task_history) == 0: # First task selection
            self.task_history.append(self.initial_task)
            available_tasks.remove(self.initial_task)
            return self.initial_task
        
        else: # Other task selection
            activations = {}
            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.detach()
                return hook
            continual_learner.avgpool.register_forward_hook(get_activation("feature maps"))
            if self.average_embedding: # compute the average embedding of all previously seen tasks
                previous_tasks_feature_maps = []
                for previous_task in self.task_history:
                    task_dataloader = self.cl_embedding_task_datasets[previous_task]
                    for x, y in task_dataloader:
                        _ = continual_learner(x.to(DEVICE))
                        previous_tasks_feature_maps.append(torch.squeeze(activations["feature maps"]).cpu())
                current_embedding = torch.mean(torch.cat(previous_tasks_feature_maps), axis=0)
            else: # compute the embedding of only the last seen task
                current_task_feature_maps = []
                last_task_dataloader = self.cl_embedding_task_datasets[self.task_history[-1]]
                for x, y in last_task_dataloader:
                    _ = continual_learner(x.to(DEVICE))
                    current_task_feature_maps.append(torch.squeeze(activations["feature maps"]).cpu())
                current_embedding = torch.mean(torch.cat(current_task_feature_maps), axis=0)

            remaining_task_embeddings = []
            for task in available_tasks:
                current_task_feature_maps = []
                for x, y in self.cl_embedding_task_datasets[task]:
                    _ = continual_learner(x.to(DEVICE))
                    current_task_feature_maps.append(torch.squeeze(activations["feature maps"]).cpu())
                remaining_task_embeddings.append(torch.mean(torch.cat(current_task_feature_maps), axis=0))
            
            distances = [cosine(current_embedding, task_embedding) for task_embedding in remaining_task_embeddings]
            if self.next_task == "min":
                chosen_task = available_tasks[np.argmin(distances)]
            elif self.next_task == "max":
                chosen_task = available_tasks[np.argmax(distances)]
            else:
                raise Exception("next task should be either min or max")
            available_tasks.remove(chosen_task)
            self.task_history.append(chosen_task)
            return chosen_task