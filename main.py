from torchvision.datasets import CIFAR10, CIFAR100
from avalanche.benchmarks.datasets import TinyImagenet
from torchvision.models import resnet18
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import numpy as np
import random

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    class_accuracy_metrics,
)
from avalanche.benchmarks.generators import nc_benchmark, data_incremental_benchmark
from avalanche.training.supervised import (
    Naive,
    EWC,
    Replay,
    AGEM,
    GSS_greedy,
    CoPE,
)

from custom_strategies.MIR_strategy import MIR
from custom_strategies.LWF_strategy import LwF
from avalanche.logging import InteractiveLogger

from curriculumdesigners.curriculumdesigner import (
    FixedCurriculumDesigner,
    RandomCurriculumDesigner,
    L2LCurriculumDesigner,
    OnTheFlyCurriculumDesigner,
)
from data.transform_util import (
    cifar10_transform,
    cifar100_transform,
    tinyimagenet_transform,
)
from data.data_util import take_subset, get_cifar100_superclass_tasks

from dataset_util.M2I_util import get_M2I_dataset

from utils.config import DEVICE
from utils.argumentparser import ArgParser
from utils.feature_tsne import feature_tsne


def train(
    lr,
    dataset_name,
    model,
    cd,
    strategy,
    num_classes,
    all_classes,
    classes_per_task,
    ewc_lambda,
    gss_n,
    memory_size,
    num_runs,
    random_seed,
    fixed_task_order=None,
    decay=0,
    plot_feature_embeddings=False,
    return_model=False,
):
    """
    Function to train and evaluate on a specific setting with specific curriculum designer.

    Args:
        lr: learning rate used during training
        dataset_name: dataset to train on
        model: backbone model
        cd: type of curriculum designer to use
        strategy: continual learning strategy to use
        num_classes: number of classes to train on
        all_classes: whether or not to train on all classes
        classes_per_task: the number of classes per increment
        ewc_lambda: lambda parameter for ewc
        memory_size: memory size for replay based methods
        num_runs: number of repetitions of the entire training configuration
        random_seed: the random seed for task creation
        fixed_task_order: the predefined fixed task order for the fixed cd, only used by oracle comparison
        plot_feature_embeddings: Whether or not to plot the feature embeddings during evaluation
        return_model: Whether or not to return the model after training, used for CKA computation

    Returns:
        dictitionary containing the following entries:
        - "accuracy": A list containing a list of the evolution of top1-accuracy for each run
        - "forgetting": A list containing a list of the evolution of forgetting for each run
        - "orderings": A list containing the orderings for the different runs
        - "complete results": A list storing the complete evaluation history of each of the different runs,
                             This list can be used to perform further investigation on the evolution of different orderings
        - "accuracy per task": A list storing the list of the average final accuracy per task for each run.
                                The order of this list is the order of tasks, so position 0 is task 0, position 1 is task 1, etc.
    """
    top1_acc_histories = []
    forgetting_histories = []
    ordering_histories = []
    result_histories = []
    accuracy_per_task_histories = []
    per_class_accuracy_histories = []

    if (
        "CIFAR10" in dataset_name and not "CIFAR100" in dataset_name
    ):  # A CIFAR10 dataset
        dataset = CIFAR10
        # Load the dataset
        embedding_set = dataset(
            "datasets",
            train=True,
            transform=cifar10_transform(resize=True),
            download=True,
        )
        train_set = dataset(
            "datasets",
            train=True,
            transform=cifar10_transform(resize=False),
            download=True,
        )
        test_set = dataset(
            "datasets",
            train=False,
            transform=cifar10_transform(resize=False),
            download=True,
        )
    elif dataset_name in ["CIFAR100", "CIFAR100-superclass"]:
        dataset = CIFAR100
        # Load the dataset
        embedding_set = dataset(
            "datasets",
            train=True,
            transform=cifar100_transform(resize=True),
            download=True,
        )
        train_set = dataset(
            "datasets",
            train=True,
            transform=cifar100_transform(resize=False),
            download=True,
        )
        test_set = dataset(
            "datasets",
            train=False,
            transform=cifar100_transform(resize=False),
            download=True,
        )
    elif dataset_name == "M2I":
        train_set = get_M2I_dataset(train=True)
        test_set = get_M2I_dataset(
            train=False, samples_per_class=50
        )  # Here we take 50 because TinyImageNet only has 50 test images per class
        embedding_set = get_M2I_dataset(
            train=True, samples_per_class=50, image_shape=(224, 224)
        )
        num_classes = 50
        classes_per_task = 10
    elif dataset_name == "TinyImageNet":
        dataset = TinyImagenet
        num_classes = 200
        classes_per_task = 5
        embedding_set = dataset(
            "datasets",
            train=True,
            transform=tinyimagenet_transform(resize=True),
            download=True,
        )
        train_set = dataset(
            "datasets",
            train=True,
            transform=tinyimagenet_transform(resize=False),
            download=True,
        )
        test_set = dataset(
            "datasets",
            train=False,
            transform=tinyimagenet_transform(resize=False),
            download=True,
        )
    else:
        raise Exception("incorrect dataset supplied/not yet implemented")

    if not all_classes:
        num_classes = num_classes
    else:
        # number of classes is equal to number of classes in dataset
        targets = train_set.targets
        if not isinstance(targets, list):
            targets = targets.tolist()
        num_classes = len(set(targets))

    if num_classes % classes_per_task != 0:
        raise Exception("number of classes should be divisible by classes per task")

    curriculum_designer = None
    model_name = model

    # Repeat for the specified number of runs
    for _ in range(num_runs):
        if model_name == "ResNet18":
            model = resnet18()
            # add classifier
            model.fc = nn.Linear(512, num_classes)
        else:
            raise Exception("incorrect model supplied/not yet implemented")

        classes = list(range(num_classes))
        # Calculate the total number of tasks
        num_tasks = num_classes // classes_per_task

        # Chop the list into sublists of length classes_per_task
        if dataset_name == "CIFAR100-superclass":
            tasks = get_cifar100_superclass_tasks()
        elif dataset_name == "CIFAR10-random":
            random.seed(random_seed)
            random.shuffle(classes)
            tasks = [classes[i : i + 2] for i in range(0, len(classes), 2)]
        else:
            tasks = [
                classes[i * classes_per_task : (i + 1) * classes_per_task]
                for i in range(num_tasks)
            ]

        task_ids = [i for i in range(len(tasks))]

        # Select the curriculum designer
        if cd == "random":
            curriculum_designer = RandomCurriculumDesigner()
        elif cd == "L2L":
            if len(tasks) > 5:
                raise Exception(
                    "L2L should not be used for task sequence longer than 5"
                )
            if curriculum_designer == None:
                curriculum_designer = L2LCurriculumDesigner(
                    dataset=embedding_set, tasks=tasks
                )
            else:
                # reset the ordering so we dont have to recompute all the values
                curriculum_designer.task_index = 0
        elif cd == "fixed":
            if fixed_task_order == None:
                raise Exception("Fixed cd should have a task order defined")
            elif len(fixed_task_order) != len(tasks):
                raise Exception("Fixed task order length incorrect")
            else:
                curriculum_designer = FixedCurriculumDesigner(order=fixed_task_order)
        elif "on the fly" in cd:
            first_task = "center"
            next_task = "max"
            if "false" in cd:
                average_embedding = False
            elif "true" in cd:
                average_embedding = True
            else:
                raise Exception(
                    "incorrect / no average embedding parameter for on the fly"
                )
            curriculum_designer = OnTheFlyCurriculumDesigner(
                dataset_name=dataset_name,
                embedding_dataset=embedding_set,
                train_set=train_set,
                tasks=tasks,
                first_task=first_task,
                next_task=next_task,
                average_embedding=average_embedding,
            )
        else:
            raise Exception("Incorrect curriculum designer provided")

        # Create the new classes benchmark to perform the experiments
        scenario = nc_benchmark(
            train_set,
            test_set,
            n_experiences=len(tasks),
            shuffle=False,
            fixed_class_order=[class_id for task in tasks for class_id in task],
            task_labels=False,
            per_exp_classes=None,
        )

        # Prepare for training & testing
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=decay)
        criterion = CrossEntropyLoss()
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(
                minibatch=True,
                epoch=True,
                epoch_running=True,
                experience=True,
                stream=True,
            ),
            forgetting_metrics(experience=True, stream=True),
            class_accuracy_metrics(stream=True),
            loggers=InteractiveLogger(),
            strict_checks=False,
        )

        # Continual learning strategy
        if strategy == "naive":
            cl_strategy = Naive(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_mb_size=32,
                train_epochs=1,
                eval_mb_size=32,
                evaluator=eval_plugin,
                device=DEVICE,
            )
        elif strategy == "ewc":
            cl_strategy = EWC(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                mode="online",
                decay_factor=0.9,
                ewc_lambda=ewc_lambda,
                train_mb_size=32,
                train_epochs=1,
                eval_mb_size=32,
                evaluator=eval_plugin,
                device=DEVICE,
            )
        elif strategy == "lwf":
            lwf_alpha = [
                (n * classes_per_task / ((n + 1) * classes_per_task))
                for n in range(num_tasks)
            ]
            cl_strategy = LwF(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                alpha=lwf_alpha,
                temperature=2,
                train_mb_size=10,
                train_epochs=1,
                eval_mb_size=32,
                evaluator=eval_plugin,
                device=DEVICE,
            )
        elif strategy == "replay":
            cl_strategy = Replay(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                mem_size=memory_size,
                train_mb_size=32,
                train_epochs=1,
                eval_mb_size=32,
                evaluator=eval_plugin,
                device=DEVICE,
            )
        elif strategy == "agem":
            cl_strategy = AGEM(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                patterns_per_exp=memory_size // len(tasks),
                sample_size=64,
                train_mb_size=32,
                train_epochs=1,
                eval_mb_size=32,
                evaluator=eval_plugin,
                device=DEVICE,
            )
        elif strategy == "mir":
            cl_strategy = MIR(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                mem_size=memory_size,
                subsample=50,
                batch_size_mem=32,
                train_mb_size=32,
                train_epochs=1,
                eval_mb_size=32,
                evaluator=eval_plugin,
                device=DEVICE,
            )
        elif strategy == "gss":
            cl_strategy = GSS_greedy(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                mem_size=memory_size,
                mem_strength=gss_n,
                input_size=train_set[0][0].shape,
                train_mb_size=32,
                train_epochs=1,
                eval_mb_size=32,
                evaluator=eval_plugin,
                device=DEVICE,
            )
        elif strategy == "cope":
            model = resnet18()
            cl_strategy = CoPE(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                mem_size=memory_size,
                p_size=model.fc.out_features,
                n_classes=num_classes,
                train_mb_size=32,
                train_epochs=1,
                eval_mb_size=32,
                evaluator=eval_plugin,
                device=DEVICE,
            )
            scenario = data_incremental_benchmark(
                scenario, shuffle=True, experience_size=32
            )
        else:
            raise Exception("Incorrect CL strategy provided")

        if strategy != "cope":  # for all strategies except CoPE
            # Get the streams from the benchmark, required for adaptive task selection
            tr_stream, ts_stream = scenario.train_stream, scenario.test_stream

            ordering = []
            result_history = []
            top1_acc_history = []
            forgetting_history = []
            per_class_accuracy = [
                [-1 for _ in range(num_classes)] for _ in range(num_tasks)
            ]
            per_task_accuracy = [
                [-1 for _ in range(num_tasks)] for _ in range(num_tasks)
            ]
            print(
                f"Training config: Dataset: {dataset_name}, Model: {model_name}, Strategy: {strategy}"
            )
            # The main training loop
            for id in range(len(task_ids)):
                print("Available tasks: {}".format(task_ids))
                if isinstance(curriculum_designer, OnTheFlyCurriculumDesigner):
                    next_task = curriculum_designer.generate_next_task(
                        task_ids, continual_learner=model
                    )
                else:
                    next_task = curriculum_designer.generate_next_task(task_ids)
                ordering.append(int(next_task))
                print("Selected task: {}".format(next_task))
                train_task = tr_stream[next_task]
                print(
                    "classes in this task: {}".format(
                        train_task.classes_in_this_experience
                    )
                )
                print("This task contains {} samples".format(len(train_task.dataset)))
                cl_strategy.train(train_task)
                evaluation_stream = [ts_stream[task_idx] for task_idx in ordering]
                evaluation_results = cl_strategy.eval(evaluation_stream)
                position_in_ordering = 0
                for key, value in evaluation_results.items():
                    if "Top1_ClassAcc_Stream/eval_phase/test_stream/Task000" in key:
                        class_id = int(key.split("/")[-1])
                        per_class_accuracy[id][class_id] = value
                    if "Top1_Acc_Exp/eval_phase" in key:
                        per_task_accuracy[id][ordering[position_in_ordering]] = value
                        position_in_ordering += 1
                top1_acc_history.append(
                    evaluation_results["Top1_Acc_Stream/eval_phase/test_stream/Task000"]
                )
                forgetting_history.append(
                    evaluation_results["StreamForgetting/eval_phase/test_stream"]
                )
                result_history.append(evaluation_results)
                if plot_feature_embeddings:
                    feature_tsne(
                        model=cl_strategy.model,
                        dataset=test_set,
                        dataset_name=dataset_name,
                        strategy=strategy,
                        tasks=tasks,
                        task_order=ordering,
                        position=len(ordering),
                    )
            top1_acc_histories.append(top1_acc_history)
            forgetting_histories.append(forgetting_history)
            ordering_histories.append(ordering)
            result_histories.append(result_history)
            per_class_accuracy_histories.append(per_class_accuracy)
            accuracy_per_task_histories.append(per_task_accuracy)
        else:  # For the CoPE strategy
            result_history = []
            top1_acc_history = []
            forgetting_history = []
            task_initial_accuracy = []
            task_evaluation_names = []
            ordering = []
            per_class_accuracy = [
                [-1 for _ in range(num_classes)] for _ in range(num_tasks)
            ]
            per_task_accuracy = [
                [-1 for _ in range(num_tasks)] for _ in range(num_tasks)
            ]

            for id in range(len(task_ids)):
                print("Available tasks: {}".format(task_ids))
                if isinstance(curriculum_designer, OnTheFlyCurriculumDesigner):
                    next_task = curriculum_designer.generate_next_task(
                        task_ids, continual_learner=model
                    )
                else:
                    next_task = curriculum_designer.generate_next_task(task_ids)
                ordering.append(int(next_task))
                print("Selected task: {}".format(next_task))

                for batch in scenario.train_stream:
                    batch_task = [
                        set(batch.classes_in_this_experience).issubset(set(task))
                        for task in tasks
                    ].index(True)
                    if (
                        batch_task == next_task
                    ):  # When the batch corresponds to the current task we train on it
                        cl_strategy.train(batch)

                print("Finished training on task {}, now evaluating".format(next_task))
                eval_stream = [
                    experience
                    for task_id, experience in enumerate(scenario.test_stream)
                    if task_id in ordering
                ]
                evaluation_results = cl_strategy.eval(eval_stream)
                for key, value in evaluation_results.items():
                    if "Top1_ClassAcc_Stream/eval_phase/test_stream/Task000" in key:
                        class_id = int(key.split("/")[-1])
                        per_class_accuracy[id][class_id] = value
                position_in_ordering = 0
                for key in evaluation_results.keys():
                    if "Top1_Acc_Exp/eval_phase" in key:
                        per_task_accuracy[id][ordering[position_in_ordering]] = (
                            evaluation_results[key]
                        )
                        position_in_ordering += 1
                if len(task_initial_accuracy) == 0:
                    for key in evaluation_results.keys():
                        if "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp" in key:
                            task_initial_accuracy.append(evaluation_results[key])
                            task_evaluation_names.append(key)
                else:
                    task_initial_accuracy.append(
                        evaluation_results[list(evaluation_results.keys())[-1]]
                    )
                    task_evaluation_names.append(list(evaluation_results.keys())[-1])
                task_forgettings = []
                for idx, task_evaluation_name in enumerate(task_evaluation_names):
                    task_forgettings.append(
                        task_initial_accuracy[idx]
                        - evaluation_results[task_evaluation_name]
                    )
                if len(task_forgettings) == 1:
                    forgetting_history.append(0.0)
                else:
                    forgetting_history.append(np.mean(task_forgettings[:-1]))
                top1_acc_history.append(
                    evaluation_results["Top1_Acc_Stream/eval_phase/test_stream/Task000"]
                )
                result_history.append(evaluation_results)
                if plot_feature_embeddings:
                    feature_tsne(
                        model=cl_strategy.model,
                        dataset=test_set,
                        dataset_name=dataset_name,
                        strategy=strategy,
                        tasks=tasks,
                        task_order=ordering,
                        position=id,
                    )

            top1_acc_histories.append(top1_acc_history)
            forgetting_histories.append(forgetting_history)
            result_histories.append(result_history)
            ordering_histories.append(ordering)
            position_in_ordering = 0
            accuracy_per_task_histories.append(per_task_accuracy)
            per_class_accuracy_histories.append(per_class_accuracy)
    if not return_model:
        return {
            "accuracy": top1_acc_histories,
            "forgetting": forgetting_histories,
            "ordering": ordering_histories,
            "complete results": result_histories,
            "accuracy per task": accuracy_per_task_histories,
            "accuracy per class": per_class_accuracy_histories,
            "tasks": tasks,
        }
    else:
        return {
            "accuracy": top1_acc_histories,
            "forgetting": forgetting_histories,
            "ordering": ordering_histories,
            "complete results": result_histories,
            "accuracy per task": accuracy_per_task_histories,
            "accuracy per class": per_class_accuracy_histories,
            "tasks": tasks,
        }, cl_strategy.model


if __name__ == "__main__":
    parser = ArgParser
    args = parser.run()
    for cd in args.cd:
        result_dict = train(
            lr=args.lr,
            dataset_name=args.dataset,
            model=args.model,
            cd=cd,
            strategy=args.strategy,
            num_classes=args.num_classes,
            all_classes=args.all_classes,
            classes_per_task=args.classes_per_task,
            ewc_lambda=args.ewc_lambda,
            gss_n=args.gss_n,
            memory_size=args.memory_size,
            num_runs=args.num_runs,
            random_seed=args.random_seed,
            fixed_task_order=args.task_order,
        )
