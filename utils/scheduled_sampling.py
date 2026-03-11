# coding=utf-8
"""
Strategy Scheduled Sampling for ESC

This module implements scheduled sampling for strategy token prediction.
Instead of always using ground-truth strategy tokens during training,
the model learns to use its own predictions with a certain probability.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def step_schedule(
    epoch: int,
    warmup_epochs: int,
    p_sample: float,
    p_random: float,
    **kwargs
) -> Tuple[float, float]:
    """
    Step schedule: fixed probabilities after warmup.

    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of warmup epochs with 100% GT
        p_sample: Probability to use predicted strategy after warmup
        p_random: Probability to use random strategy after warmup

    Returns:
        (p_sample, p_random) for current epoch
    """
    if epoch < warmup_epochs:
        return 0.0, 0.0
    return p_sample, p_random


def linear_schedule(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    p_sample_max: float,
    p_random_max: float = 0.0,
    **kwargs
) -> Tuple[float, float]:
    """
    Linear schedule: gradually increase replacement probability.

    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of warmup epochs with 100% GT
        total_epochs: Total number of training epochs
        p_sample_max: Maximum probability for predicted strategy
        p_random_max: Maximum probability for random strategy

    Returns:
        (p_sample, p_random) for current epoch
    """
    if epoch < warmup_epochs:
        return 0.0, 0.0

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
    progress = min(1.0, progress)

    p_sample = p_sample_max * progress
    p_random = p_random_max * progress

    return p_sample, p_random


def inverse_sigmoid_schedule(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    p_sample_max: float,
    p_random_max: float = 0.0,
    k: float = 5.0,
    **kwargs
) -> Tuple[float, float]:
    """
    Inverse sigmoid schedule: smooth transition with sigmoid decay.

    The probability increases slowly at first, then faster, then slowly again.

    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of warmup epochs with 100% GT
        total_epochs: Total number of training epochs
        p_sample_max: Maximum probability for predicted strategy
        p_random_max: Maximum probability for random strategy
        k: Steepness of the sigmoid curve

    Returns:
        (p_sample, p_random) for current epoch
    """
    if epoch < warmup_epochs:
        return 0.0, 0.0

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
    progress = min(1.0, progress)

    # Sigmoid transformation: map [0, 1] progress to [0, 1] with S-curve
    # Using inverse sigmoid decay: starts slow, accelerates, then slow
    x = k * (2 * progress - 1)  # Map to [-k, k]
    sigmoid_progress = 1 / (1 + math.exp(-x))
    # Normalize to [0, 1]
    sigmoid_progress = (sigmoid_progress - 1 / (1 + math.exp(k))) / (1 / (1 + math.exp(-k)) - 1 / (1 + math.exp(k)))

    p_sample = p_sample_max * sigmoid_progress
    p_random = p_random_max * sigmoid_progress

    return p_sample, p_random


SCHEDULE_FUNCTIONS = {
    'step': step_schedule,
    'linear': linear_schedule,
    'inverse_sigmoid': inverse_sigmoid_schedule,
}


class StrategyScheduledSampler:
    """
    Manages scheduled sampling for strategy tokens during training.

    During training, instead of always using ground-truth strategy tokens,
    this sampler probabilistically replaces them with:
    - Model's predicted strategy (p_sample)
    - Random strategy (p_random)
    - Ground-truth strategy (1 - p_sample - p_random)

    This helps bridge the gap between teacher forcing and inference.
    """

    def __init__(
        self,
        schedule_type: Literal['step', 'linear', 'inverse_sigmoid'] = 'step',
        warmup_epochs: int = 1,
        total_epochs: int = 10,
        p_sample: float = 0.1,
        p_random: float = 0.1,
        num_strategies: int = 8,
        strategy_token_offset: int = 54944,
    ):
        """
        Initialize the scheduled sampler.

        Args:
            schedule_type: Type of schedule function ('step', 'linear', 'inverse_sigmoid')
            warmup_epochs: Number of epochs with 100% ground-truth
            total_epochs: Total number of training epochs
            p_sample: Probability to use predicted strategy (for step schedule)
                     or maximum probability (for linear/sigmoid schedules)
            p_random: Probability to use random strategy
            num_strategies: Number of strategy tokens (default: 8)
            strategy_token_offset: Vocabulary offset for strategy tokens
        """
        self.schedule_type = schedule_type
        self.schedule_fn = SCHEDULE_FUNCTIONS[schedule_type]
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.p_sample_config = p_sample
        self.p_random_config = p_random
        self.num_strategies = num_strategies
        self.strategy_token_offset = strategy_token_offset

        self.current_epoch = 0
        self.current_p_sample = 0.0
        self.current_p_random = 0.0

    def set_epoch(self, epoch: int):
        """Update the current epoch and recalculate probabilities."""
        self.current_epoch = epoch
        self.current_p_sample, self.current_p_random = self.schedule_fn(
            epoch=epoch,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.total_epochs,
            p_sample=self.p_sample_config,
            p_random=self.p_random_config,
            p_sample_max=self.p_sample_config,
            p_random_max=self.p_random_config,
        )

    def get_current_probs(self) -> Tuple[float, float, float]:
        """
        Get current probabilities.

        Returns:
            (p_gt, p_sample, p_random)
        """
        p_gt = 1.0 - self.current_p_sample - self.current_p_random
        return p_gt, self.current_p_sample, self.current_p_random

    def sample_strategy_tokens(
        self,
        gt_strategy_ids: torch.Tensor,
        strategy_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample strategy tokens based on current schedule.

        For each sample in the batch:
        - With probability p_gt: keep ground-truth
        - With probability p_sample: use predicted strategy (argmax of logits)
        - With probability p_random: use random strategy

        Args:
            gt_strategy_ids: Ground-truth strategy IDs [batch_size], values in [0, num_strategies-1]
            strategy_logits: Model's strategy prediction logits [batch_size, num_strategies]
                           Required if p_sample > 0

        Returns:
            Sampled strategy IDs [batch_size], values in [0, num_strategies-1]
        """
        batch_size = gt_strategy_ids.size(0)
        device = gt_strategy_ids.device

        p_gt, p_sample, p_random = self.get_current_probs()

        # If in warmup or no replacement, return GT
        if p_sample == 0.0 and p_random == 0.0:
            return gt_strategy_ids.clone()

        # Generate random values for each sample
        rand_vals = torch.rand(batch_size, device=device)

        # Start with GT
        result = gt_strategy_ids.clone()

        # Apply predicted strategy where rand_vals in [p_gt, p_gt + p_sample)
        if p_sample > 0.0 and strategy_logits is not None:
            pred_strategy_ids = torch.argmax(strategy_logits, dim=-1)
            use_predicted = (rand_vals >= p_gt) & (rand_vals < p_gt + p_sample)
            result = torch.where(use_predicted, pred_strategy_ids, result)

        # Apply random strategy where rand_vals >= p_gt + p_sample
        if p_random > 0.0:
            random_strategy_ids = torch.randint(
                0, self.num_strategies, (batch_size,), device=device
            )
            use_random = rand_vals >= (p_gt + p_sample)
            result = torch.where(use_random, random_strategy_ids, result)

        return result

    def __repr__(self) -> str:
        p_gt, p_sample, p_random = self.get_current_probs()
        return (
            f"StrategyScheduledSampler("
            f"schedule={self.schedule_type}, "
            f"epoch={self.current_epoch}, "
            f"p_gt={p_gt:.2f}, p_sample={p_sample:.2f}, p_random={p_random:.2f})"
        )
