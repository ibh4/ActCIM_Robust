from __future__ import annotations

import torch
import torch.nn as nn


def register_activation_hooks(model):
    activations = {}

    def hook_fn(name):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                activations[name] = out.detach().clone()
            elif isinstance(out, (list, tuple)) and isinstance(out[0], torch.Tensor):
                activations[name] = out[0].detach().clone()
        return fn

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            h = module.register_forward_hook(hook_fn(name))
            handles.append(h)

    return activations, handles


def collect_layer_activations(model, data_loader, device, max_batches=50):
    activations_collected = {}

    activation_storage, handles = register_activation_hooks(model)

    def hook_fn(name):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                if name not in activations_collected:
                    activations_collected[name] = []
                activations_collected[name].append(out.detach().cpu())
        return fn

    for h in handles:
        h.remove()

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            h = module.register_forward_hook(hook_fn(name))
            handles.append(h)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)
            _ = model(inputs)

    for h in handles:
        h.remove()

    stacked_activations = {}
    for name, acts in activations_collected.items():
        if acts:
            stacked_activations[name] = torch.cat(acts, dim=0)

    return stacked_activations
