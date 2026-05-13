from __future__ import annotations

import timm
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet50,
)


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def configure_trainable_layers(model_name: str, model: nn.Module, freeze_backbone: bool = True) -> None:
    if not freeze_backbone:
        unfreeze_module(model)
        return

    freeze_module(model)
    if model_name == "efficientnet_b0":
        unfreeze_module(model.classifier)
    elif model_name == "resnet50":
        unfreeze_module(model.fc)
    elif model_name == "convnext_tiny":
        unfreeze_module(model.classifier)
    elif model_name == "swin_tiny":
        unfreeze_module(model.head)


def unfreeze_last_blocks(model_name: str, model: nn.Module) -> None:
    if model_name == "efficientnet_b0":
        unfreeze_module(model.features[-2:])
        unfreeze_module(model.classifier)
    elif model_name == "resnet50":
        unfreeze_module(model.layer4)
        unfreeze_module(model.fc)
    elif model_name == "convnext_tiny":
        unfreeze_module(model.features[-1])
        unfreeze_module(model.classifier)
    elif model_name == "swin_tiny":
        unfreeze_module(model.layers[-1])
        unfreeze_module(model.head)
    else:
        unfreeze_module(model)


def build_model(model_name: str, num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif model_name == "swin_tiny":
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    configure_trainable_layers(model_name, model, freeze_backbone=freeze_backbone)
    return model
