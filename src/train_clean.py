import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from src.utils import accuracy

def train_clean(model, train_loader, test_loader, device,
                epochs=20, lr=0.1, momentum=0.9, weight_decay=5e-4,
                optimizer_name="sgd", ckpt_path=None):
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
        for x, y in tqdm(train_loader, desc=f"Clean Epoch {ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

        scheduler.step()
        acc = accuracy(model, test_loader, device)
        print(f"[CLEAN] Epoch {ep:02d}/{epochs} | Test Clean: {acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if ckpt_path and acc > best:
            best = acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved checkpoint: {ckpt_path}")

    return best
