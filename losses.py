import torch.nn as nn
import torch
import torch.nn.functional as F


def deepinfo_loss(real, fake):
    logits_real = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
    logits_fake = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))

    # shape: [num_imgs, 1]
    loss = logits_real + logits_fake
    return loss


def pair_wise_loss(x, y, delta_v = 0.5, delta_d = 1.5):
    """
    :param x: [b, c]
    :param y: [b, c]
    :return: pair_wise_loss, float.
    """
    b, c, _, _ = x.shape
    center = (x + y) / 2  # [b, c, 1, 1]
    # dist = 0.5 * ((x - center).norm(dim=1) + (y - center).norm(dim=1))
    # loss = 0.5 * lambd * (torch.max(dist, torch.ones_like(dist))**2).mean()
    # dist = 0.5 * (((x - center) ** 2).sum(dim=1) + ((y - center) ** 2).sum(dim=1))
    # loss = 0.5 * lambd * dist.mean()

    L_var = 0.5 * ((torch.max((x - center).norm(dim=1) - delta_v, torch.Tensor([0]).to(x.device)) ** 2).mean() + \
                   (torch.max((y - center).norm(dim=1) - delta_v, torch.Tensor([0]).to(y.device)) ** 2).mean())

    # [b, 1, c, 1] - [1, b, c, 1]
    delta_matrix = (delta_d * (torch.ones([b, b, 1, 1]) - torch.eye(b).unsqueeze(2).unsqueeze(3))).to(x.device)
    L_dist = (torch.max(2 * delta_matrix - (center.permute(0, 2, 1, 3) - center.permute(2, 0, 1, 3)).norm(dim=2),
                        torch.Tensor([0]).to(x.device)) ** 2).mean()

    L_reg = center.norm(dim=1).mean()

    return L_var, L_dist, L_reg


def kl_divergence(z_mu, z_logvar, lambd=0.01):
    # Only caculate the KL divergence over channel.
    kld = - 0.5 * (1 + z_logvar - z_mu ** 2 - torch.exp(z_logvar)).mean()
    loss = lambd * kld

    return loss
