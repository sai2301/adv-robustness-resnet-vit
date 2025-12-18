import argparse
import yaml
import torch

from src.data import get_cifar10_loaders
from src.models import get_model
from src.train_clean import train_clean
from src.train_pgd_at import train_pgd_at
from src.train_trades import train_trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Config:", cfg)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=int(cfg["batch_size"]),
        image_size=int(cfg["image_size"])
    )

    model = get_model(
        name=str(cfg["model"]),
        num_classes=10,
        pretrained=bool(cfg.get("pretrained", False))
    ).to(device)

    mode = str(cfg.get("mode", "clean")).lower()

    epochs = int(cfg["epochs"])
    lr = float(cfg["lr"])
    momentum = float(cfg.get("momentum", 0.9))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    optimizer_name = str(cfg.get("optimizer", "sgd"))
    ckpt_path = cfg.get("ckpt_path", None)

    if mode == "clean":
        print("Running Clean Training")
        best_acc = train_clean(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            ckpt_path=ckpt_path
        )
        print(f"Best test accuracy: {best_acc*100:.2f}%")

    elif mode == "pgd_at":
        print("Running PGD Adversarial Training (PGD-AT)")
        eps = float(cfg["eps"])
        alpha = float(cfg.get("alpha", 2/255))
        pgd_steps = int(cfg.get("pgd_steps", 10))

        best_acc = train_pgd_at(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            eps=eps,
            alpha=alpha,
            pgd_steps=pgd_steps,
            ckpt_path=ckpt_path
        )
        print(f"Best robust (PGD) accuracy: {best_acc*100:.2f}%")

    elif mode == "trades":
        print("Running TRADES Training")
        eps = float(cfg["eps"])
        alpha = float(cfg.get("alpha", 2/255))
        trades_steps = int(cfg.get("trades_steps", 10))
        beta = float(cfg.get("beta", 6.0))

        best_acc = train_trades(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            eps=eps,
            alpha=alpha,
            trades_steps=trades_steps,
            beta=beta,
            ckpt_path=ckpt_path
        )
        print(f"Best robust (PGD) accuracy: {best_acc*100:.2f}%")

    else:
        raise ValueError(f"Unknown mode: {mode}. Use clean / pgd_at / trades")


if __name__ == "__main__":
    main()
