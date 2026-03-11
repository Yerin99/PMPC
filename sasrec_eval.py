# coding=utf-8
"""
SASREC Evaluation Script.

Evaluates SASREC model and dumps probability distributions for ensemble.

Following ESCMamba's implementation.

Usage:
    python sasrec_eval.py \
        --model_path ./DATA/sasrec/best_model.pt \
        --eval_file ./_reformat/valid.txt \
        --output_file ./DATA/sasrec/valid_probs.json
"""

import argparse
import json
import logging
import os

import torch
import torch.nn.functional as F
import numpy as np

from sasrec.model import SASREC
from sasrec.dataset import (
    create_turn_level_dataloader,
    NUM_TOKENS,
    ID_TO_STRATEGY,
    STRATEGY_OFFSET,
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SASREC model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained SASREC model')
    parser.add_argument('--eval_file', type=str, required=True,
                        help='Path to evaluation data file')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save probability dump (optional)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=50,
                        help='Maximum sequence length')

    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> SASREC:
    """Load trained SASREC model."""
    checkpoint = torch.load(model_path, map_location=device)

    # Get model args
    model_args = checkpoint.get('args', {})

    # Following ESCMamba: output_num_items = num_items = 10
    model = SASREC(
        num_items=NUM_TOKENS,  # 10
        hidden_size=model_args.get('hidden_size', 256),
        num_blocks=model_args.get('num_blocks', 4),
        num_heads=model_args.get('num_heads', 4),
        max_len=model_args.get('max_len', 50),
        dropout_rate=model_args.get('dropout', 0.2),
        output_num_items=NUM_TOKENS,  # 10 (same as input vocab)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f'Loaded model from {model_path}')
    logger.info(f'Model config: hidden_size={model_args.get("hidden_size", 256)}, '
                f'num_blocks={model_args.get("num_blocks", 4)}, '
                f'num_heads={model_args.get("num_heads", 4)}')

    if 'valid_acc' in checkpoint:
        logger.info(f'Checkpoint valid accuracy: {checkpoint["valid_acc"]:.4f}')

    return model


def evaluate_and_dump(
    model: SASREC,
    dataloader,
    device: torch.device,
    output_file: str = None,
) -> dict:
    """
    Evaluate model and optionally dump probabilities.

    Returns:
        dict with evaluation metrics and optionally probability dumps
    """
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []
    all_dialog_indices = []
    all_turn_indices = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            actual_lens = batch['actual_lens'].to(device)  # Actual sequence lengths
            targets = batch['targets'].to(device)  # IDs 2~9

            # Get predictions (using actual last position, following ESCMamba)
            probs = model.predict(input_ids, attention_mask, actual_lens)  # (batch_size, 10)
            preds = probs.argmax(dim=-1)  # (batch_size,) - IDs 0~9

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_dialog_indices.extend(batch['dialog_indices'])
            all_turn_indices.extend(batch['turn_indices'])

    # Concatenate
    all_probs = torch.cat(all_probs, dim=0)  # (num_samples, 10)
    all_preds = torch.cat(all_preds, dim=0)  # (num_samples,)
    all_targets = torch.cat(all_targets, dim=0)  # (num_samples,) - IDs 2~9

    # Calculate metrics (targets and preds are both in the same ID space now)
    accuracy = (all_preds == all_targets).float().mean().item()

    # Top-3 accuracy
    _, top3_preds = all_probs.topk(3, dim=-1)
    top3_correct = (top3_preds == all_targets.unsqueeze(-1)).any(dim=-1)
    top3_accuracy = top3_correct.float().mean().item()

    logger.info(f'Evaluation Results:')
    logger.info(f'  Top-1 Accuracy: {accuracy:.4f}')
    logger.info(f'  Top-3 Accuracy: {top3_accuracy:.4f}')
    logger.info(f'  Total samples: {len(all_targets)}')

    # Per-strategy accuracy (strategies are IDs 2~9)
    logger.info(f'Per-strategy accuracy:')
    for strategy_id in range(STRATEGY_OFFSET, NUM_TOKENS):
        mask = all_targets == strategy_id
        if mask.sum() > 0:
            strat_acc = (all_preds[mask] == all_targets[mask]).float().mean().item()
            strat_name = ID_TO_STRATEGY.get(strategy_id, f'Strategy_{strategy_id}')
            logger.info(f'  {strat_name}: {strat_acc:.4f} (n={mask.sum().item()})')

    results = {
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'num_samples': len(all_targets),
    }

    # Dump probabilities if output file specified
    if output_file is not None:
        prob_dump = []

        for i in range(len(all_targets)):
            # Extract only strategy probabilities (indices 2~9) for ensemble
            strategy_probs = all_probs[i, STRATEGY_OFFSET:].numpy().tolist()  # 8 values

            prob_dump.append({
                'dialog_idx': all_dialog_indices[i],
                'turn_idx': all_turn_indices[i],
                'target': all_targets[i].item() - STRATEGY_OFFSET,  # Convert to 0~7 for ensemble
                'pred': all_preds[i].item() - STRATEGY_OFFSET if all_preds[i].item() >= STRATEGY_OFFSET else -1,
                'probs': strategy_probs,  # 8 probabilities for strategies
            })

        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(prob_dump, f, indent=2)

        logger.info(f'Probabilities saved to {output_file}')

        results['output_file'] = output_file

    return results


def main():
    args = parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load model
    model = load_model(args.model_path, device)

    # Create dataloader
    logger.info(f'Loading evaluation data from {args.eval_file}')
    dataloader = create_turn_level_dataloader(
        args.eval_file,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    logger.info(f'Evaluation samples: {len(dataloader.dataset)}')

    # Evaluate
    results = evaluate_and_dump(
        model,
        dataloader,
        device,
        output_file=args.output_file,
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"SASREC Evaluation Summary")
    print(f"{'='*50}")
    print(f"Top-1 Accuracy: {results['accuracy']:.4f}")
    print(f"Top-3 Accuracy: {results['top3_accuracy']:.4f}")
    print(f"Total samples: {results['num_samples']}")
    if args.output_file:
        print(f"Probabilities saved to: {args.output_file}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
