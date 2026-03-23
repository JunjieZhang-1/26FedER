from torchvision import transforms
from torchvision.datasets import MNIST  # 👈 新增这一行
from .cifar import CIFAR10, CIFAR100


def load_dataset(dataset):
    """
    Returns: dataset_train, dataset_test, num_classes
    """
    dataset_train = None
    dataset_test = None
    num_classes = 0

    if dataset == 'cifar10':
        trans_cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_cifar10_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = CIFAR10(
            root='./data/cifar',
            download=True,
            train=True,
            transform=trans_cifar10_train,
        )
        dataset_test = CIFAR10(
            root='./data/cifar',
            download=True,
            train=False,
            transform=trans_cifar10_val,
        )
        num_classes = 10

    elif dataset == 'cifar100':
        trans_cifar100_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])],
        )
        trans_cifar100_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])],
        )
        dataset_train = CIFAR100(
            root='./data/cifar100',
            download=True,
            train=True,
            transform=trans_cifar100_train,
        )
        dataset_test = CIFAR100(
            root='./data/cifar100',
            download=True,
            train=False,
            transform=trans_cifar100_val,
        )
        num_classes = 100
    elif dataset == 'mnist':
        from torchvision.datasets import MNIST

        # 新增：强制替换为稳定的 AWS 镜像链接，解决 404 报错
        MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
             'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
             'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
             '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
             'ec29112dd5afa0611ce80d1b7f02629c')
        ]

        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = MNIST(
            root='./data/mnist',
            download=True,
            train=True,
            transform=trans_mnist,
        )

        # # 👇 新的暴力截断法：直接修改底层张量，保留所有原生属性
        # dataset_train.data = dataset_train.data[:6000]
        # dataset_train.targets = dataset_train.targets[:6000]

        dataset_test = MNIST(
            root='./data/mnist',
            download=True,
            train=False,
            transform=trans_mnist,
        )
        num_classes = 10

    else:
        raise NotImplementedError('Error: unrecognized dataset')

    return dataset_train, dataset_test, num_classes
