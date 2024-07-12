from utils.argumentparser import ArgParser
from itertools import permutations
from collections import defaultdict
from main import train

import random
import numpy as np
import json


def get_rank(dictionary, key, descending=True):
    """
    Function to return the rank of item "key" in dictionary "dictionary"

    Args:
    -  dictionary: The dictionary storing the score for each key
    -  key: The key of which we are interested in the rank
    -  descending: Whether or not to sort the values descendingly

    Returns:
    -  rank: the rank of the key (rank 1 is best)
    """
    # Sort the dictionary items based on values in descending order
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=descending)

    # Find the index of the key in the sorted list
    try:
        rank = sorted_items.index((key, dictionary[key])) + 1
        return rank
    except ValueError:
        # Handle the case where the key is not found in the dictionary
        return None


def get_orderings(num_tasks, count_first_last=5):
    """
    Function to return a list of random permutations subject to the following conditions:
    1. Every task is the first task exactly "count_first_last" times
    2. Every task is the last task exactly "count_first_last" times
    3. Every combination of first task - last task occurs at most once

    Args:
    num_tasks: the number of tasks in the dataset
    count_first_last: how often each task should be the first and last task

    Returns:
    orderings: A list of permutatations subject to the three conditions listed above
    """
    count_last_dict = defaultdict(int)
    orderings = []
    random.seed(42)
    tasks = list(range(num_tasks))

    for element in tasks:
        last_task_candidates = [
            item
            for item in tasks
            if item != element
            and count_last_dict[item] == min(count_last_dict.values())
        ]
        if (
            len(last_task_candidates) < 5
        ):  # The current element has been selected less than the other elements, but it cannot be the last element as well, so we also consider all elements that occur once more
            last_task_candidates = [
                item
                for item in tasks
                if item != element
                and count_last_dict[item] <= min(count_last_dict.values()) + 1
            ]
        last_tasks = random.sample(last_task_candidates, count_first_last)
        for last_task in last_tasks:
            count_last_dict[last_task] += 1
            other_tasks = [
                task for task in tasks if task != element and task != last_task
            ]
            random.shuffle(other_tasks)
            orderings.append([element] + other_tasks + [last_task])
    assert (
        len(orderings) == num_tasks * count_first_last
    ), "number of orderings should be equal to num_tasks * count_first_last"
    return orderings


def oracle_comparison(args, strategies):
    """
    Code for running the oracle comparison experiment
    There are 2 different scenarios for this experiment
    1. The dataset has 5 tasks or less
    2. The dataset has more than 5 tasks

    In scenario 1 this experiment runs all possible permutations on the specified strategies

    In scenario 2 this experiment creates a set of random orderings with the following constraints
    Constraint 1: Every task is the first task exactly "x" times
    Constraint 2: Every task is the last task exactly "x" times
    Constraint 3 :Every combination of first task - last task occurs at most once
    This results in x * num_tasks permutations, which are all tested on the specified strategies

    In both scenarios the following 5 metrics are logged for the evaluated permutations:
    1. Mean final average accuracy
    2. Standard deviation final average accuracy
    3. Mean forgetting
    4. Standard deviation forgetting
    5. Mean final accuracy per task

    Args:
        - args: The commandline arguments to specify the experiments
        - strategies: The strategies to evaluate
    """

    num_tasks = args.num_classes // args.classes_per_task

    if args.dataset == "M2I":
        num_tasks = 5

    if args.dataset not in [
        "MNIST",
        "CIFAR10",
        "CIFAR10-random",
        "CIFAR100",
        "CIFAR100-superclass",
        "CIFAR100-M2I-like",
        "M2I",
        "CIFAR10-fixed-replay-bad",
        "CIFAR10-fixed-replay-good",
        "CIFAR10-fixed-regularization",
    ]:
        raise Exception(
            f"Oracle comparison not currently supported for dataset {args.dataset}"
        )

    if "replay" in args.dataset:
        strategies = ["replay", "mir", "gss", "cope"]
    elif args.dataset == "CIFAR10-fixed-regularization":
        strategies = ["ewc", "lwf", "agem"]

    if num_tasks <= 5:
        orderings = list(permutations(range(num_tasks)))
    else:
        orderings = get_orderings(num_tasks)

    for strategy in strategies:
        print("Oracle Comparison strategy: ", strategy)
        accuracy_dict = dict()
        accuracy_std_dict = dict()
        forgetting_dict = dict()
        forgetting_std_dict = dict()
        accuracy_per_task_dict = dict()
        for ordering in orderings:
            results = train(
                lr=args.lr,
                dataset_name=args.dataset,
                model=args.model,
                cd="fixed",
                strategy=strategy,
                num_classes=args.num_classes,
                all_classes=args.all_classes,
                classes_per_task=args.classes_per_task,
                ewc_lambda=args.ewc_lambda,
                gss_n=args.gss_n,
                memory_size=args.memory_size,
                num_runs=args.num_runs,
                fixed_task_order=ordering,
                random_seed=args.random_seed,
            )
            accuracy_dict[str(ordering)] = np.mean(results["accuracy"], axis=0)[-1]
            accuracy_std_dict[str(ordering)] = np.std(results["accuracy"], axis=0)[-1]
            forgetting_dict[str(ordering)] = np.mean(results["forgetting"], axis=0)[-1]
            forgetting_std_dict[str(ordering)] = np.std(results["forgetting"], axis=0)[
                -1
            ]
            accuracy_per_task_dict[str(ordering)] = np.mean(
                [result[-1] for result in results["accuracy per task"]], axis=0
            ).tolist()

        with open(
            f"logs/oracle_comparison/strategy-{strategy}-dataset-{args.dataset}-accuracy.json",
            "w",
        ) as file:
            json.dump(accuracy_dict, file)
        with open(
            f"logs/oracle_comparison/strategy-{strategy}-dataset-{args.dataset}-accuracy_std.json",
            "w",
        ) as file:
            json.dump(accuracy_std_dict, file)
        with open(
            f"logs/oracle_comparison/strategy-{strategy}-dataset-{args.dataset}-forgetting.json",
            "w",
        ) as file:
            json.dump(forgetting_dict, file)
        with open(
            f"logs/oracle_comparison/strategy-{strategy}-dataset-{args.dataset}-forgetting_std.json",
            "w",
        ) as file:
            json.dump(forgetting_std_dict, file)
        with open(
            f"logs/oracle_comparison/strategy-{strategy}-dataset-{args.dataset}-accuracy_per_task.json",
            "w",
        ) as file:
            json.dump(accuracy_per_task_dict, file)


if __name__ == "__main__":
    parser = ArgParser
    args = parser.run()
    strategies = ["replay", "gss", "mir", "cope"]
    print("Starting Oracle Comparison")
    oracle_comparison(args, strategies=strategies)
