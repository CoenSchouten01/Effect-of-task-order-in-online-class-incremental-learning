from torchvision import transforms


def expand_channels(x):
    return x.expand(3, -1, -1)


def mnist_transform(resize=False, resize_size=(224, 224)):
    """
    The transformation for the MNIST dataset, this function reshapes the images,
    transforms the input to tensors, normalizes the values using
    the mean and std (from: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151?permalink_comment_id=2851662#gistcomment-2851662)
    and additionally expands the number of channels from 1 to 3.

    Returns:
        A composed transformation for the MNIST dataset
    """
    if resize:
        return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(expand_channels),
        ]
    )
    else:
        return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(expand_channels),
        ]
    )

def fashion_mnist_transform(resize=False, resize_size=(224, 224)):
    """
    The transformation for the FashionMNist dataset, this function reshapes the images,
    transforms the input to tensors, normalizes the values using
    the mean and std (from: https://github.com/lifelonglab/M2I_I2M_benchmark/blob/main/scenarios/datasets/fashion_mnist.py)
    and additionally expands the number of channels from 1 to 3.

    Returns:
        A composed transformation for the FashionMNIST dataset
    """
    if resize:
        return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(expand_channels),
        ]
    )
    else:
        return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(expand_channels),
        ]
    )

def svhn_transform(resize=False, resize_size=(224, 224)):
    """
    The transformation for the SVHN dataset, this function transforms the input to tensorsand possibly reshapes

    Returns:
        A composed transformation for the SVHN dataset
    """
    if resize:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.ToTensor()
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

def tinyimagenet_transform(resize=False, resize_size=(224, 224)):
    if resize:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )


def cifar10_transform(resize=False, resize_size=(224, 224)):
    """
    The transformation for the CIFAR10 dataset, this function reshapes the images,
    transforms the input to tensors and normalizes the values,
    using the mean and std. (from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151?permalink_comment_id=2851662#gistcomment-2851662)

    Returns:
        A composed transformation for the CIFAR10 dataset
    """
    if resize:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )


def cifar100_transform(resize=False, resize_size=(224, 224)):
    """
    The transformation for the CIFAR100 dataset, this function reshapes the images,
    transforms the input to tensors and normalizes the values,
    using the mean and std. (from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151?permalink_comment_id=2851662#gistcomment-2851662)

    Returns:
        A composed transformation for the CIFAR100 dataset
    """
    if resize:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]
        )
    else: 
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]
        )