import torch
import torch.nn.functional as F

def fgsm_attack(model, x, y, eps=8/255):
    """
    x is in [0,1] pixel space.
    model includes NormalizeLayer internally.
    """
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + eps * grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv

def pgd_attack(model, x, y, eps=8/255, alpha=2/255, steps=10, random_start=True):
    """
    Standard Linf PGD.
    """
    model.eval()
    x_orig = x.clone().detach()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    else:
        x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()

        # project back to Linf ball
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def trades_pgd_attack(model, x, eps=8/255, alpha=2/255, steps=10, random_start=True):
    """
    TRADES adversary maximizes KL( model(x_adv) || model(x_clean) ).
    We generate x_adv using KL divergence objective.
    """
    model.eval()
    x_orig = x.detach()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    else:
        x_adv = x_orig.clone()

    with torch.no_grad():
        logits_clean = model(x_orig)
        p_clean = F.softmax(logits_clean, dim=1)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits_adv = model(x_adv)
        logp_adv = F.log_softmax(logits_adv, dim=1)

        # KL(p_clean || p_adv) = sum p_clean * (log p_clean - log p_adv)
        loss_kl = F.kl_div(logp_adv, p_clean, reduction="batchmean")
        grad = torch.autograd.grad(loss_kl, x_adv)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()
