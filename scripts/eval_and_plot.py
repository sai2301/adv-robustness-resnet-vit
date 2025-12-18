import os
import argparse
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.data import get_cifar10_loaders
from src.models import get_model
from src.attacks import fgsm_attack, pgd_attack


@torch.no_grad()
def eval_clean(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def eval_fgsm(model, loader, device, eps=8/255):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = fgsm_attack(model, x, y, eps=eps)
        with torch.no_grad():
            logits = model(x_adv)
            pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def eval_pgd(model, loader, device, eps=8/255, alpha=2/255, steps=10):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True)
        with torch.no_grad():
            logits = model(x_adv)
            pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="List of yaml config files")
    ap.add_argument("--out_dir", default="outputs/results_csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    rows = []

    for cfg_path in args.configs:
        cfg = yaml.safe_load(open(cfg_path, "r"))

        model_name = cfg["model"]
        image_size = int(cfg["image_size"])
        batch_size = int(cfg["batch_size"])
        ckpt_path = cfg["ckpt_path"]

        eps = float(cfg.get("eps", 8/255))
        alpha = float(cfg.get("alpha", 2/255))
        pgd_steps = int(cfg.get("pgd_steps", 10))

        print("\nEvaluating:", cfg_path)
        print("  ckpt:", ckpt_path)

        _, test_loader = get_cifar10_loaders(batch_size=batch_size, image_size=image_size)

        model = get_model(
            name=model_name,
            num_classes=10,
            pretrained=cfg.get("pretrained", False)
        ).to(device)

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        clean_acc = eval_clean(model, test_loader, device)
        fgsm_acc = eval_fgsm(model, test_loader, device, eps=eps)
        pgd_acc  = eval_pgd(model, test_loader, device, eps=eps, alpha=alpha, steps=pgd_steps)

        rows.append({
            "config": os.path.basename(cfg_path),
            "model": model_name,
            "mode": cfg.get("mode", "clean"),
            "clean_acc": clean_acc,
            "fgsm_acc": fgsm_acc,
            "pgd_acc": pgd_acc
        })

        print(f"  Clean: {clean_acc*100:.2f}% | FGSM: {fgsm_acc*100:.2f}% | PGD: {pgd_acc*100:.2f}%")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print("\nSaved:", csv_path)

    # Plot: Clean vs Robust
    plt.figure()
    x = range(len(df))
    plt.bar([i - 0.2 for i in x], df["clean_acc"], width=0.2, label="Clean")
    plt.bar([i for i in x],       df["fgsm_acc"],  width=0.2, label="FGSM")
    plt.bar([i + 0.2 for i in x], df["pgd_acc"],   width=0.2, label="PGD")
    plt.xticks(list(x), df["model"] + "-" + df["mode"], rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 Accuracy under Attacks")
    plt.legend()
    plt.tight_layout()
    plot1 = "outputs/plots/robustness_bar.png"
    plt.savefig(plot1)
    print("Saved:", plot1)

    # Plot: PGD only
    plt.figure()
    plt.bar(df["model"] + "-" + df["mode"], df["pgd_acc"])
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("PGD Accuracy")
    plt.title("PGD Robust Accuracy Comparison")
    plt.tight_layout()
    plot2 = "outputs/plots/pgd_only.png"
    plt.savefig(plot2)
    print("Saved:", plot2)


if __name__ == "__main__":
    main()
