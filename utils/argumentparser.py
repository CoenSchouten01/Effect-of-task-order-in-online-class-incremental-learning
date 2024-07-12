import argparse


class ArgParser:
    def __init__(self):
        pass

    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="learning rate (default value: 0.001)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            choices=["CIFAR10", "CIFAR10-random", "CIFAR100", "CIFAR100-superclass", "M2I", "TinyImageNet"],
            default="CIFAR10",
            help="The dataset to train on (default value: CIFAR10)",
        )
        parser.add_argument(
            "--model",
            type=str,
            choices=["ResNet18"],
            default="ResNet18",
            help="The network architecture (default value: ResNet18)",
        )
        parser.add_argument(
            "--cd",
            type=str,
            nargs="+",
            choices=[
                "random",
                "L2L",
                "fixed",
                "on the fly-center-max-false",
                "on the fly-center-max-true"
            ],
            default=["random"],
            help="select the curriculum designers (default value: random), it is possible to select multiple curriculum designers at once",
        )
        parser.add_argument(
            "--strategy",
            type=str,
            choices=["ewc", "lwf", "replay", "agem", "mir", "gss" ,"cope"],
            default="ewc",
            help="selects a CL strategy (default value: ewc)",
        )
        parser.add_argument(
            "--num_classes",
            type=int,
            default=10,
            help="select the number of classes to train on (default value: 10)",
        )
        parser.add_argument(
            "--all_classes",
            type=bool,
            default=False,
            help="whether or not to train on all classes (default value: False)",
        )
        parser.add_argument(
            "--classes_per_task",
            type=int,
            default=2,
            help="the number of classes per increment (default value: 2)",
        )
        parser.add_argument(
            "--ewc_lambda",
            type=float,
            choices=[0.1, 1, 100, 1000],
            default=100,
            help="select lambda for EWC (default value: 100)",
        )
        parser.add_argument(
            "--gss_n",
            type=int,
            choices=[1, 10, 20, 50],
            default=10,
            help="number of gradient vectors drawn from the memory buffer to compute cosine similarity to"
        )
        parser.add_argument(
            "--memory_size",
            type=int,
            choices=[1000, 5000, 10000],
            default=5000,
            help="select memory size for replay methods (default value: 1000)",
        )
        parser.add_argument(
            "--num_runs",
            type=int,
            default=1,
            help="set the number of repeated runs (default value: 1)",
        )
        parser.add_argument(
            "--random_seed",
            type=int,
            default=42,
            help="The random seed for the experiments",
        )
        parser.add_argument(
            "--task_order",
            type=int,
            nargs="+",
            default=[0, 1, 2, 3, 4],
            help="fixed task order for fixed CD"
        )
        args = parser.parse_args()
        return args
