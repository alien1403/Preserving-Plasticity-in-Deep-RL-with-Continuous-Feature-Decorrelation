from typing import Dict
import torch
import torch.nn as nn
from .metrics import (
    compute_effective_rank,
    compute_stable_rank,
    compute_dormant_neurons,
    compute_dead_relus,
    compute_weight_metrics,
)

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

    def compute_diagnostics(self, clear_after: bool = True) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {}
        for name, acts in self.activations.items():
            if acts is None or acts.numel() == 0: continue
            dormant_ratio, _ = compute_dormant_neurons(acts, tau=self.tau)
            diagnostics[f"{name}/effective_rank"] = compute_effective_rank(acts)
            diagnostics[f"{name}/stable_rank"] = compute_stable_rank(acts)
            diagnostics[f"{name}/dormant_ratio"] = dormant_ratio
            diagnostics[f"{name}/dead_relus"] = compute_dead_relus(acts)
        for name, module in self.model.named_modules():
            if name in self.activations:
                w_stats = compute_weight_metrics(module)
                for k, v in w_stats.items(): diagnostics[f"{name}/{k}"] = v
        if clear_after: self.clear()
        return diagnostics

    def clear(self) -> None: self.activations.clear()
    def remove_hooks(self) -> None:
        for handle in self.hooks: handle.remove()
        self.hooks.clear()
        self.clear()
    def __del__(self): self.remove_hooks()
