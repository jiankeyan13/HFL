from  registry import register_dataset
from dataset_store import DatasetStore
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import os

def foo(flag:bool):
    return "train" if flag else "test"
@register_dataset('cifar10')
def build_cifar10(root, is_train):    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if not os.path.exists(root):
        os.makedirs(root)

    if is_train:
        real_dataset = CIFAR10(root=root, train=is_train, download=True, transform = train_transform)
    else:
        real_dataset = CIFAR10(root=root, train=is_train, download=True, transform = test_transform)
    return DatasetStore('cifar10', foo(is_train), real_dataset)