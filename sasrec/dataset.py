# coding=utf-8
"""
Strategy Sequence Dataset for SASREC training.

Extracts strategy sequences from ESConv dialogues for next-strategy prediction.

Token indexing (following ESCMamba):
- 0: [PAD]
- 1: [no_history]
- 2~9: Strategies

Labels use the same indexing (2~9), with ignore_index=0 for padding.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional


# Strategy mapping (same order as in the main model)
STRATEGY_LIST = [
    'Question',
    'Restatement or Paraphrasing',
    'Reflection of feelings',
    'Self-disclosure',
    'Affirmation and Reassurance',
    'Providing Suggestions',
    'Information',
    'Others',
]

# Token indexing: 0=PAD, 1=[no_history], 2~9=strategies
PAD_TOKEN_ID = 0
NO_HISTORY_TOKEN_ID = 1
STRATEGY_OFFSET = 2  # strategies start from index 2

STRATEGY_TO_ID = {strat: i + STRATEGY_OFFSET for i, strat in enumerate(STRATEGY_LIST)}
ID_TO_STRATEGY = {v: k for k, v in STRATEGY_TO_ID.items()}
ID_TO_STRATEGY[PAD_TOKEN_ID] = '[PAD]'
ID_TO_STRATEGY[NO_HISTORY_TOKEN_ID] = '[no_history]'

NUM_STRATEGIES = len(STRATEGY_LIST)  # 8
NUM_TOKENS = NUM_STRATEGIES + STRATEGY_OFFSET  # 10 (PAD + no_history + 8 strategies)


def extract_strategy_sequences(data_file: str) -> List[List[int]]:
    """
    Extract strategy sequences from ESConv dialogue file.

    Args:
        data_file: path to train.txt/valid.txt/test.txt

    Returns:
        List of strategy sequences (each is a list of strategy IDs, using 2~9)
    """
    sequences = []

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            dialog = data.get('dialog', [])

            # Extract strategies from sys turns
            strategy_seq = []
            for turn in dialog:
                if turn.get('speaker') == 'sys' and 'strategy' in turn:
                    strategy_name = turn['strategy']
                    if strategy_name in STRATEGY_TO_ID:
                        strategy_id = STRATEGY_TO_ID[strategy_name]
                        strategy_seq.append(strategy_id)

            # Only add sequences with at least 1 strategy
            if len(strategy_seq) >= 1:
                sequences.append(strategy_seq)

    return sequences


def extract_turn_level_samples(data_file: str) -> List[Dict]:
    """
    Extract turn-level samples for evaluation alignment with BlenderBot.

    Each sample corresponds to one sys turn (same as BlenderBot's inference).

    Args:
        data_file: path to train.txt/valid.txt/test.txt

    Returns:
        List of dicts with:
            - history: list of previous strategy IDs (can be empty for first turn)
            - target: target strategy ID for this turn (2~9)
            - dialog_idx: dialogue index
            - turn_idx: turn index within dialogue
    """
    samples = []
    dialog_idx = 0

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            dialog = data.get('dialog', [])

            # Collect sys turns with their strategies
            strategy_history = []
            turn_idx = 0

            for i, turn in enumerate(dialog):
                if turn.get('speaker') == 'sys' and 'strategy' in turn:
                    strategy_name = turn['strategy']
                    if strategy_name in STRATEGY_TO_ID:
                        strategy_id = STRATEGY_TO_ID[strategy_name]

                        # Skip first turn (i=0) to match BlenderBot's inputter
                        # BlenderBot's strat.py line 91: if i > 0 and dialog[i]['speaker'] == 'sys'
                        if i > 0:
                            samples.append({
                                'history': strategy_history.copy(),
                                'target': strategy_id,  # Keep as 2~9 (following ESCMamba)
                                'dialog_idx': dialog_idx,
                                'turn_idx': turn_idx,
                            })

                        strategy_history.append(strategy_id)
                        turn_idx += 1

            dialog_idx += 1

    return samples


class StrategySequenceDataset(Dataset):
    """
    Dataset for SASREC training on strategy sequences.

    Following ESCMamba's approach:
    - One sample per dialogue (full sequence)
    - Input: [no_history] + strategies[:-1]
    - Target: strategies (IDs 2~9, same as input vocab)
    - ignore_index=0 (PAD) for CrossEntropyLoss
    """

    def __init__(
        self,
        data_file: str,
        max_len: int = 50,
        training: bool = True,
    ):
        """
        Args:
            data_file: path to dialogue file
            max_len: maximum sequence length
            training: if True, shuffle during training; if False, no shuffle
        """
        self.max_len = max_len
        self.training = training

        # Extract raw sequences (strategy IDs are 2~9)
        self.raw_sequences = extract_strategy_sequences(data_file)

        # Generate samples (one per dialogue)
        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Tuple[List[int], List[int]]]:
        """
        Generate training samples following ESCMamba's approach.

        For each dialogue with strategies [s1, s2, s3, s4]:
        - Input: [no_history, s1, s2, s3]
        - Target: [s1, s2, s3, s4]  (IDs 2~9, NOT converted to 0~7)
        """
        samples = []

        for seq in self.raw_sequences:
            if len(seq) < 1:
                continue

            # Input: [no_history] + all strategies except last
            # Target: all strategies (keep IDs as 2~9, following ESCMamba)
            input_seq = [NO_HISTORY_TOKEN_ID] + seq[:-1]
            target_seq = seq  # targets are strategy IDs (2~9) - NO offset!

            # Truncate to max_len (keep recent)
            if len(input_seq) > self.max_len:
                input_seq = input_seq[-self.max_len:]
                target_seq = target_seq[-self.max_len:]

            samples.append((input_seq, target_seq))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_seq, target_seq = self.samples[idx]

        # Pad to max_len (ESCMamba style - FIXED padding)
        pad_len = self.max_len - len(input_seq)
        input_ids = input_seq + [0] * pad_len
        target_ids = target_seq + [0] * pad_len

        # Attention mask (ESCMamba style)
        attention_mask = [1 if x != 0 else 0 for x in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


class TurnLevelDataset(Dataset):
    """
    Dataset for turn-level evaluation, aligned with BlenderBot's output.

    Each sample is one sys turn with its strategy history.
    Following ESCMamba: prepend [no_history] to the input sequence.
    """

    def __init__(
        self,
        data_file: str,
        max_len: int = 50,
    ):
        self.max_len = max_len
        self.samples = extract_turn_level_samples(data_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        history = sample['history']  # list of strategy IDs (2~9)
        target = sample['target']  # strategy ID (2~9)

        # Prepend [no_history] token (following ESCMamba)
        input_seq = [NO_HISTORY_TOKEN_ID] + history

        # Truncate to max_len (keep recent)
        if len(input_seq) > self.max_len:
            input_seq = input_seq[-self.max_len:]

        # Store actual length before padding (needed for correct prediction)
        actual_len = len(input_seq)

        # Pad to max_len (ESCMamba style - FIXED padding)
        pad_len = self.max_len - len(input_seq)
        input_ids = input_seq + [0] * pad_len

        # Attention mask (ESCMamba style)
        attention_mask = [1 if x != 0 else 0 for x in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'actual_len': actual_len,  # For correct last-position prediction
            'target': torch.tensor(target, dtype=torch.long),  # Keep as 2~9!
            'dialog_idx': sample['dialog_idx'],
            'turn_idx': sample['turn_idx'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for StrategySequenceDataset.
    Since __getitem__ now returns fixed-length padded tensors (ESCMamba style),
    we just stack them.
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'target_ids': torch.stack([item['target_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
    }


def collate_fn_turn_level(batch: List[Dict]) -> Dict:
    """
    Collate function for TurnLevelDataset.
    Since __getitem__ now returns fixed-length padded tensors (ESCMamba style),
    we just stack them.
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'actual_lens': torch.tensor([item['actual_len'] for item in batch], dtype=torch.long),
        'targets': torch.stack([item['target'] for item in batch]),
        'dialog_indices': [item['dialog_idx'] for item in batch],
        'turn_indices': [item['turn_idx'] for item in batch],
    }


def create_dataloader(
    data_file: str,
    batch_size: int = 128,
    max_len: int = 50,
    training: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for SASREC training/evaluation."""
    dataset = StrategySequenceDataset(data_file, max_len, training)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_turn_level_dataloader(
    data_file: str,
    batch_size: int = 128,
    max_len: int = 50,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for turn-level evaluation."""
    dataset = TurnLevelDataset(data_file, max_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_turn_level,
        num_workers=num_workers,
        pin_memory=True,
    )
