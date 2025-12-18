import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.attacks import pgd_attack
from src.utils import accuracy

def pgd_accuracy(model, loader, device, eps=8/255, alpha=2/255, steps=10):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)

def train_pgd_at(model, train_loader, test_loader, device,
                 epochs=20, lr=0.1, momentum=0.9, weight_decay=5e-4,
                 optimizer_name="sgd", eps=8/255, alpha=2/255, pgd_steps=10, ckpt_path=None):
    model.to(device)
    ce = nn.CrossEntropyLoss()

    opt = optimizer_name.lower()
    if opt == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best = 0.0

    for ep in range(1, epochs+1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"PGD-AT Epoch {ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)

            with torch.enable_grad():
                x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=pgd_steps, random_start=True)

            optimizer.zero_grad()
            logits = model(x_adv)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

        scheduler.step()
        clean = accuracy(model, test_loader, device)
        robust = pgd_accuracy(model, test_loader, device, eps=eps, alpha=alpha, steps=10)
        print(f"[PGD-AT] Epoch {ep:02d}/{epochs} | Test Clean: {clean:.4f} | Test PGD: {robust:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if ckpt_path and robust > best:
            best = robust
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved robust checkpoint: {ckpt_path}")
    return best
