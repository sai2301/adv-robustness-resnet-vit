from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, image_size=32, num_workers=2):
    tf_train = []
    tf_test  = []

    if image_size != 32:
        tf_train += [transforms.Resize((image_size, image_size))]
        tf_test  += [transforms.Resize((image_size, image_size))]

    tf_train += [
        transforms.RandomCrop(image_size, padding=4) if image_size == 32 else transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    tf_test += [
        transforms.ToTensor(),
    ]

    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.Compose(tf_train))
    test_ds  = datasets.CIFAR10(root="data", train=False, download=True, transform=transforms.Compose(tf_test))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader  = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader
