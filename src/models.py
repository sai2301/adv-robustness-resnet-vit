import torch
import torch.nn as nn
import torchvision.models as tvm
import timm

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

class NormalizeLayer(nn.Module):
    def __init__(self, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std  = torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std

def get_model(name="resnet18", num_classes=10, pretrained=False):
    name = name.lower()

    if name == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model: {name}. Use 'resnet18' or 'vit'.")

    # normalize inside model so adversarial attacks operate on raw [0,1] pixel space
    model = nn.Sequential(NormalizeLayer(), model)
    return model
