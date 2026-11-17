from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from looktwice.plotting import SEQ_CMAP


def gradcam(model: torch.nn.Module, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)
        feats = model.forward_features(x)
        logit = model.forward_head(feats).squeeze(1)
        sign = torch.where(logit >= 0, 1.0, -1.0)
        grads = torch.autograd.grad((logit * sign).sum(), feats)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    flat = cam.flatten(1)
    lo, hi = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
    cam = (cam - lo) / (hi - lo).clamp_min(1e-8)
    return cam.detach().cpu().numpy(), torch.sigmoid(logit).detach().cpu().numpy()


def denormalize(x: torch.Tensor, mean, std) -> np.ndarray:
    img = x.detach().cpu().numpy().transpose(1, 2, 0) * np.asarray(std) + np.asarray(mean)
    return np.clip(img, 0, 1)


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    heat = SEQ_CMAP(cam)[..., :3]
    a = alpha * cam[..., None]
    return np.clip((1 - a) * img + a * heat, 0, 1)
