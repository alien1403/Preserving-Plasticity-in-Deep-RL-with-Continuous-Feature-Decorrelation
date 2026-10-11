from typing import Dict
import torch
import torch.nn as nn

class DiagnosticHookManager:
    def __init__(self, model: nn.Module, target_layer_types: tuple = (nn.Linear, nn.Conv2d), tau: float = 0.05):
        self.model = model
        self.target_layer_types = target_layer_types
        self.tau = tau
        self.hooks = []
        self.activations: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, self.target_layer_types):
                def hook_fn(layer_name: str):
                    def hook(mod, inp, out):
                        self.activations[layer_name] = out.detach()
                    return hook
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)

    def clear(self) -> None:
        self.activations.clear()

    def remove_hooks(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.clear()

    def __del__(self):
        self.remove_hooks()
