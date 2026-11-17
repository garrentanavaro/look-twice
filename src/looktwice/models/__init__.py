from __future__ import annotations

import timm
import torch.nn as nn

MODEL_ZOO = {
    "resnet50": "resnet50.tv_in1k",
    "xception": "legacy_xception.tf_in1k",
    "efficientnet_b4": "efficientnet_b4.ra2_in1k",
}


def build_model(name: str, pretrained: bool = True, drop_rate: float = 0.0,
                freeze_backbone: bool = False) -> nn.Module:
    model = timm.create_model(MODEL_ZOO.get(name, name), pretrained=pretrained,
                              num_classes=1, drop_rate=drop_rate)
    if freeze_backbone:
        head = model.get_classifier()
        head_params = {id(p) for p in head.parameters()}
        for p in model.parameters():
            p.requires_grad = id(p) in head_params
    return model


def load_checkpoint(path, device="cpu") -> tuple[nn.Module, dict]:
    import torch
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    meta = {k: v for k, v in ckpt.items() if k != "model"}
    return model.to(device).eval(), meta


def model_data_config(model: nn.Module) -> dict:
    return timm.data.resolve_model_data_config(model)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
