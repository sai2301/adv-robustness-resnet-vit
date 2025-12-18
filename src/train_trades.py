"""
TRADES Adversarial Training for CIFAR-10 models.

Assumptions:
- Inputs from dataloader are in pixel space [0,1]
- Model applies normalization internally (NormalizeLayer in src/models.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.attacks import trades_pgd_attack


@torch.no_grad()
def evaluate_clean(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def evaluate_pgd(model, dataloader, device, eps=8/255, alpha=2/255, steps=10):
    from src.attacks import pgd_attack

    model.eval()
    correct, total = 0, 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        with torch.enable_grad():
            x_adv = pgd_attack(
                model, x, y,
                eps=eps, alpha=alpha, steps=steps,
                random_start=True
            )

        with torch.no_grad():
            logits = model(x_adv)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def train_trades(
    model,
    train_loader,
    test_loader,
    device,
    epochs=20,
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    optimizer_name="sgd",
    eps=8/255,
    alpha=2/255,
    trades_steps=10,
    beta=6.0,
    ckpt_path=None
):
    ce = nn.CrossEntropyLoss()

    opt = optimizer_name.lower()
    if opt == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == "sgd":
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer_name must be 'sgd' or 'adamw'")

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_robust = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in tqdm(train_loader, desc=f"TRADES Epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)

            # 1) TRADES adversarial examples
            x_adv = trades_pgd_attack(
                model, x,
                eps=eps, alpha=alpha, steps=trades_steps,
                random_start=True
            )

            # 2) TRADES loss
            optimizer.zero_grad()

            logits_clean = model(x)
            logits_adv = model(x_adv)

            loss_clean = ce(logits_clean, y)

            p_clean = F.softmax(logits_clean, dim=1).detach()
            logp_adv = F.log_softmax(logits_adv, dim=1)
            loss_robust = F.kl_div(logp_adv, p_clean, reduction="batchmean")

            loss = loss_clean + beta * loss_robust
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        scheduler.step()

        test_clean = evaluate_clean(model, test_loader, device)
        test_pgd = evaluate_pgd(model, test_loader, device, eps=eps, alpha=alpha, steps=10)

        print(
            f"[TRADES] Epoch {epoch:02d}/{epochs} | "
            f"Loss: {total_loss/total:.4f} | "
            f"Test Clean: {test_clean:.4f} | "
            f"Test PGD: {test_pgd:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if ckpt_path is not None and test_pgd > best_robust:
            best_robust = test_pgd
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved robust checkpoint: {ckpt_path}")

    return best_robust
