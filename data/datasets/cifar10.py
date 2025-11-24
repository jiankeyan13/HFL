from registry import register_dataset
from dataset_store import DatasetStore
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

aug_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def _build_cifar10_impl(root, is_train, use_aug):
    # 根据是否增强选择 transform
    t = aug_transform if (is_train and use_aug) else base_transform
    
    # 下载/加载
    real_dataset = CIFAR10(root=root, train=is_train, download=True, transform=t)
    
    if is_train:
        split = "train" if use_aug else "train_plain"
    else:
        split = "test"
        
    return DatasetStore("cifar10", split, real_dataset)

# === 注册三个具体版本 ===

# 1. 训练用（带增强） -> 'cifar10_train_aug'
@register_dataset('cifar10_train_aug')
def build_cifar10_aug(root, is_train):
    # 强制 is_train=True, use_aug=True
    return _build_cifar10_impl(root, is_train=True, use_aug=True)

# 2. 训练集切分出的验证/测试用（无增强） -> 'cifar10_train_plain'
@register_dataset('cifar10_train_plain')
def build_cifar10_train_plain(root, is_train):
    # 强制 is_train=True, 但 use_aug=False
    return _build_cifar10_impl(root, is_train=True, use_aug=False)

# 3. 全局测试集（无增强） -> 'cifar10_test_plain'
@register_dataset('cifar10_test_plain')
def build_cifar10_test(root, is_train):
    # 强制 is_train=False, use_aug=False
    return _build_cifar10_impl(root, is_train=False, use_aug=False)
