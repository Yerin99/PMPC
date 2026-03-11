# coding=utf-8
"""
Grid Search for 018 Ensemble Weights (Joint Probe + SASREC).

Finds optimal alpha on validation set:
  p_final = alpha * p_probe + (1-alpha) * p_sasrec

Usage:
    python -m ensemble.grid_search_018 \
        --probe_probs_file ./DATA/.../gen.json \
        --sasrec_probs_file ./DATA/sasrec/valid_probs.json \
        --output_file ./DATA/ensemble_018/grid_search_results.json
"""

import argparse
import json
import logging
import os

import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Grid search for 018 ensemble weights')

    parser.add_argument('--probe_probs_file', type=str, required=True,
                        help='Path to probe gen.json with pred_strat_id_dist')
    parser.add_argument('--sasrec_probs_file', type=str, required=True,
                        help='Path to SASREC probability dump JSON')
    parser.add_argument('--output_file', type=str, default='./DATA/ensemble_018/grid_search_results.json',
                        help='Path to save grid search results')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Step size for alpha grid search (default: 0.05)')

    return parser.parse_args()


def load_probe_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = []
    for item in data:
        if 'pred_strat_id_dist' not in item:
            continue
        prob_str = item['pred_strat_id_dist']
        prob = np.array([float(x) for x in prob_str.split()])
        results.append({
            'prob': prob,
            'pred': item.get('pred_strat_id', int(prob.argmax())),
            'sample_id': item.get('sample_id', -1),
        })
    return results


def load_sasrec_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = []
    for item in data:
        results.append({
            'prob': np.array(item['probs']),
            'pred': item['pred'],
            'target': item['target'],
        })
    return results


def compute_ensemble_accuracy(probe_probs, sasrec_probs, targets, alpha):
    ensemble_probs = alpha * probe_probs + (1 - alpha) * sasrec_probs
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

    preds = ensemble_probs.argmax(axis=1)
    accuracy = (preds == targets).mean()

    top3_preds = np.argsort(ensemble_probs, axis=1)[:, -3:]
    top3_correct = np.any(top3_preds == targets.reshape(-1, 1), axis=1)
    top3_accuracy = top3_correct.mean()

    return {
        'accuracy': float(accuracy),
        'top3_accuracy': float(top3_accuracy),
        'alpha': float(alpha),
    }


def main():
    args = parse_args()

    logger.info(f"Loading probe data from {args.probe_probs_file}")
    probe_data = load_probe_data(args.probe_probs_file)
    logger.info(f"Loaded {len(probe_data)} probe samples")

    logger.info(f"Loading SASREC data from {args.sasrec_probs_file}")
    sasrec_data = load_sasrec_data(args.sasrec_probs_file)
    logger.info(f"Loaded {len(sasrec_data)} SASREC samples")

    min_len = min(len(probe_data), len(sasrec_data))
    if len(probe_data) != len(sasrec_data):
        logger.warning(f"Sample count mismatch: probe={len(probe_data)}, SASREC={len(sasrec_data)}")
    probe_data = probe_data[:min_len]
    sasrec_data = sasrec_data[:min_len]

    probe_probs = np.stack([item['prob'] for item in probe_data])
    sasrec_probs = np.stack([item['prob'] for item in sasrec_data])
    targets = np.array([item['target'] for item in sasrec_data])

    # Baselines
    probe_only = compute_ensemble_accuracy(probe_probs, sasrec_probs, targets, 1.0)
    sasrec_only = compute_ensemble_accuracy(probe_probs, sasrec_probs, targets, 0.0)

    logger.info(f"\n{'='*50}")
    logger.info(f"Baseline Results:")
    logger.info(f"  Probe only:  acc={probe_only['accuracy']:.4f}, top3={probe_only['top3_accuracy']:.4f}")
    logger.info(f"  SASREC only: acc={sasrec_only['accuracy']:.4f}, top3={sasrec_only['top3_accuracy']:.4f}")
    logger.info(f"{'='*50}\n")

    # Grid search
    best_result = None
    best_accuracy = -1
    all_results = []

    alphas = np.arange(0, 1 + args.step, args.step)
    for a in alphas:
        result = compute_ensemble_accuracy(probe_probs, sasrec_probs, targets, a)
        all_results.append(result)
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_result = result
        logger.info(f"alpha={a:.2f}: acc={result['accuracy']:.4f}, top3={result['top3_accuracy']:.4f}")

    logger.info(f"\n{'='*50}")
    logger.info(f"Best: alpha={best_result['alpha']:.2f}, acc={best_result['accuracy']:.4f}")
    logger.info(f"{'='*50}\n")

    output = {
        'baseline': {'probe_only': probe_only, 'sasrec_only': sasrec_only},
        'best': best_result,
        'all_results': all_results,
        'config': {
            'probe_probs_file': args.probe_probs_file,
            'sasrec_probs_file': args.sasrec_probs_file,
            'step': args.step,
            'num_samples': int(min_len),
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {args.output_file}")

    print(f"\n{'='*60}")
    print(f"018 ENSEMBLE GRID SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"Probe only:     acc={probe_only['accuracy']:.4f}")
    print(f"SASREC only:    acc={sasrec_only['accuracy']:.4f}")
    print(f"Best Ensemble:  acc={best_result['accuracy']:.4f} (alpha={best_result['alpha']:.2f})")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
