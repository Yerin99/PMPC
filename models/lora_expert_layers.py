# coding=utf-8

from contextlib import contextmanager
from typing import Callable, List, Optional

import torch
import torch.nn as nn
try:
    from transformers.modeling_utils import Conv1D
except ImportError:
    Conv1D = None  # Not available in this transformers version; GPT-2 Conv1D unused for BlenderBot


class LoRAExpertContext:
    def __init__(self):
        self.expert_ids: Optional[torch.Tensor] = None
        self.fixed_expert_id: Optional[int] = None
        self.disable_lora: bool = False


class LoRAWrapperBase(nn.Module):
    def __init__(
        self,
        base_module: nn.Module,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        dropout: float,
        num_experts: Optional[int] = None,
        context: Optional[LoRAExpertContext] = None,
    ):
        super().__init__()
        self.base = base_module
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.num_experts = num_experts
        self.context = context
        self.dropout = nn.Dropout(dropout)

        if num_experts is None:
            self.lora_a = nn.Parameter(torch.empty(rank, in_features))
            self.lora_b = nn.Parameter(torch.empty(out_features, rank))
        else:
            self.lora_a = nn.Parameter(torch.empty(num_experts, rank, in_features))
            self.lora_b = nn.Parameter(torch.empty(num_experts, out_features, rank))
        self.reset_parameters()

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def reset_parameters(self):
        nn.init.normal_(self.lora_a, std=0.02)
        nn.init.zeros_(self.lora_b)

    def _forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        inter = torch.einsum("...i,ri->...r", x, self.lora_a)
        delta = torch.einsum("...r,or->...o", inter, self.lora_b)
        return delta

    def _forward_expert(self, x: torch.Tensor) -> torch.Tensor:
        if self.context is None:
            raise RuntimeError("LoRA expert context is not set")
        if self.context.disable_lora:
            return x.new_zeros(x.shape[:-1] + (self.out_features,))
        if self.context.expert_ids is not None:
            expert_ids = self.context.expert_ids
            if expert_ids.dim() != 1 or expert_ids.size(0) != x.size(0):
                raise RuntimeError("expert_ids must be [batch] and match input batch size")
            lora_a = self.lora_a[expert_ids]
            lora_b = self.lora_b[expert_ids]
            inter = torch.einsum("b...i,bri->b...r", x, lora_a)
            delta = torch.einsum("b...r,bor->b...o", inter, lora_b)
            return delta

        if self.context.fixed_expert_id is None:
            raise RuntimeError("Either expert_ids or fixed_expert_id must be set for expert LoRA")

        expert_id = int(self.context.fixed_expert_id)
        lora_a = self.lora_a[expert_id]
        lora_b = self.lora_b[expert_id]
        inter = torch.einsum("...i,ri->...r", x, lora_a)
        delta = torch.einsum("...r,or->...o", inter, lora_b)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        dropped = self.dropout(x)
        if self.num_experts is None:
            delta = self._forward_shared(dropped)
        else:
            delta = self._forward_expert(dropped)
        return base_out + delta * self.scaling


class LoRALinear(LoRAWrapperBase):
    def __init__(
        self,
        base_module: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float,
        num_experts: Optional[int] = None,
        context: Optional[LoRAExpertContext] = None,
    ):
        super().__init__(
            base_module=base_module,
            in_features=base_module.in_features,
            out_features=base_module.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_experts=num_experts,
            context=context,
        )


class LoRAConv1D(LoRAWrapperBase):
    def __init__(
        self,
        base_module: Conv1D,
        rank: int,
        alpha: int,
        dropout: float,
        num_experts: Optional[int] = None,
        context: Optional[LoRAExpertContext] = None,
    ):
        super().__init__(
            base_module=base_module,
            in_features=base_module.weight.size(0),
            out_features=base_module.weight.size(1),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_experts=num_experts,
            context=context,
        )
        self.nf = base_module.nf


def get_submodule(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_submodule(root: nn.Module, path: str, module: nn.Module):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def wrap_modules_with_lora(
    root: nn.Module,
    predicate: Callable[[str, nn.Module], bool],
    rank: int,
    alpha: int,
    dropout: float,
    num_experts: Optional[int] = None,
    context: Optional[LoRAExpertContext] = None,
) -> List[str]:
    target_names: List[str] = []
    for name, module in root.named_modules():
        if not name:
            continue
        if predicate(name, module):
            target_names.append(name)

    for name in target_names:
        module = get_submodule(root, name)
        if isinstance(module, LoRAWrapperBase):
            continue
        if isinstance(module, nn.Linear):
            wrapped = LoRALinear(
                base_module=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                num_experts=num_experts,
                context=context,
            )
        elif isinstance(module, Conv1D):
            wrapped = LoRAConv1D(
                base_module=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                num_experts=num_experts,
                context=context,
            )
        else:
            continue
        set_submodule(root, name, wrapped)
    return target_names


@contextmanager
def disable_expert_lora(model):
    patched = []
    for m in model.modules():
        if isinstance(m, LoRAWrapperBase) and m.num_experts is not None:
            orig_forward = m.forward
            base_module = m.base
            m.forward = lambda x, _base=base_module: _base(x)
            patched.append((m, orig_forward))
    try:
        yield
    finally:
        for m, orig_forward in patched:
            m.forward = orig_forward
