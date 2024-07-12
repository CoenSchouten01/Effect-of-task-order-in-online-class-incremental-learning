from main import train
from utils.argumentparser import ArgParser

import json
import numpy as np
import matplotlib.pyplot as plt


def comparison_cd(args, cds, plot_only):
    """
    Experiment returning plots comparing the performance of different cds on average accuracy and average forgetting metrics

    Args:
        args: the command line arguments used for the training loop
        plot_only: whether or not to only update the plot based on the logs, if true, only plot, else, run the training
    """
    top1_acc_histories = dict()
    forgetting_histories = dict()
    if plot_only:
        for cd in cds:
            with open(
                f"logs/cd_comparison/{cd}-strategy-{args.strategy}-dataset-{args.dataset}-{args.num_classes}-{args.all_classes}-{args.classes_per_task}.json",
                "r",
            ) as file:
                results = json.load(file)
            top1_acc_histories.update({cd: results["accuracy"]})
            forgetting_histories.update({cd: results["forgetting"]})
    else:
        for cd in cds:
            if cd == "random":
                results = train(
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
                    num_runs=10,
                    random_seed=args.random_seed,
                )
            else:
                results = train(
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
                )
            top1_acc_cd = {cd: results["accuracy"]}
            forgetting_cd = {cd: results["forgetting"]}
            top1_acc_histories.update(top1_acc_cd)
            forgetting_histories.update(forgetting_cd)
            if args.memory_size != 1000:
                with open(
                    f"logs/cd_comparison/{cd}-strategy-{args.strategy}-dataset-{args.dataset}-{args.num_classes}-{args.all_classes}-{args.classes_per_task}-{args.memory_size}.json",
                    "w",
                ) as file:
                    json.dump(results, file)
            else:
                with open(
                    f"logs/cd_comparison/{cd}-strategy-{args.strategy}-dataset-{args.dataset}-{args.num_classes}-{args.all_classes}-{args.classes_per_task}.json",
                    "w",
                ) as file:
                    json.dump(results, file)

    plot_comparison(
        top1_acc_histories,
        f"Comparison of average accuracy of different CDs \n dataset: {args.dataset} strategy: {args.strategy} classes per task: {args.classes_per_task}",
        f"strategy-{args.strategy}-dataset-{args.dataset}-{args.num_classes}-{args.all_classes}-{args.classes_per_task}-acc",
        "avg accuracy",
    )
    plot_comparison(
        forgetting_histories,
        f"Comparison of average forgetting of different CDs \n dataset: {args.dataset} strategy: {args.strategy} classes per task: {args.classes_per_task}",
        f"strategy-{args.strategy}-dataset-{args.dataset}-{args.num_classes}-{args.all_classes}-{args.classes_per_task}-forgetting",
        "avg forgetting",
    )


def plot_comparison(result_dict, title, file_name, y_label):
    """
    Plot the results of experiment comparing performance of different cds
    stored in result_dict and save the plot.

    Args:
        result_dict: dictionary containing the metric we want to plot for each CD
        title: the title of the plot, also used as file name
        y_label: the label for the y-axis
    """
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.xlabel("Task number")
    plt.ylabel(y_label)
    for cd in result_dict.keys():
        mean = np.mean(result_dict[cd], axis=0)
        std = np.std(result_dict[cd], axis=0)
        plt.plot(mean, label=cd)
        plt.fill_between(
            x=np.arange(len(mean)), y1=mean - std, y2=mean + std, alpha=0.2
        )
    plt.xticks(np.arange(len(mean)))
    # plt.legend()
    plt.legend(loc=(1.01, 0.01), ncol=2)
    plt.tight_layout()
    plt.savefig("figures/cd_comparison/" + file_name)


if __name__ == "__main__":
    parser = ArgParser
    args = parser.run()
    print("Starting CD Comparison")
    if args.dataset in ["CIFAR10", "CIFAR10-random", "M2I"]:
        cds = [
            "on the fly-center-max-false",
            "on the fly-center-max-true",
            "L2L",
        ]
    else:
        cds = [
            "on the fly-center-max-false",
            "on the fly-center-max-true",
        ]
    comparison_cd(args, cds=cds, plot_only=False)
