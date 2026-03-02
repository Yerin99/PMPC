# PMPC: Prompt Learning with Multiperspective Cues for Emotional Support Conversation Systems

Reproduction of the following paper:

> Yangyang Xu, Zhuoer Zhao, Xiao Sun, and Xun Yang. **"Prompt Learning With Multiperspective Cues for Emotional Support Conversation Systems."** *IEEE Transactions on Computational Social Systems*, 2025. [[paper]](https://doi.org/10.1109/TCSS.2025.3539915)

```bib
@article{xu2025pmpc,
  title={Prompt Learning With Multiperspective Cues for Emotional Support Conversation Systems},
  author={Xu, Yangyang and Zhao, Zhuoer and Sun, Xiao and Yang, Xun},
  journal={IEEE Transactions on Computational Social Systems},
  year={2025},
  doi={10.1109/TCSS.2025.3539915}
}
```

## Overview

PMPC extracts **multiperspective cues** from dialogue history and constructs soft prompts to guide a BlenderBot-small backbone for emotional support response generation.

**Cues Catcher** (Section III-A):
- **Topic Cues (Ht)**: GloVe + PMI-based keyword extraction
- **Prior Knowledge Cues (He)**: DPR retrieval from training responses
- **Mental State Cues (Hc)**: COMET-ATOMIC 2020 (6 user + 3 listener relations)
- **Other Cues**: Post (Hp), Situation (Hs) from encoder hidden states

**Prompt Builder** (Section III-B):
- Semantic enhancement prompt (Pe) for encoder
- Semantic constraint prompt (Pd) for decoder, conditioned on strategy prediction

## Base Code

This implementation builds upon the ESConv codebase:

> Siyang Liu\*, Chujie Zheng\*, et al. **"Towards Emotional Support Dialog Systems."** *ACL 2021.* [[paper]](https://arxiv.org/abs/2106.01144) [[repo]](https://github.com/thu-coai/Emotional-Support-Conversation)

## Setup

### Environment

```bash
conda create -n pmpc python=3.9
conda activate pmpc
pip install torch transformers nltk numpy tqdm
```

### External Models

Download and place under `external_models/`:

| Model | Source | Path |
|-------|--------|------|
| COMET-ATOMIC 2020 | [HuggingFace](https://huggingface.co/mismayil/comet-bart-ai2) | `external_models/comet-atomic-2020/` |
| DPR Context Encoder | [HuggingFace](https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base) | `external_models/dpr-ctx-encoder/` |
| DPR Question Encoder | [HuggingFace](https://huggingface.co/facebook/dpr-question_encoder-single-nq-base) | `external_models/dpr-question-encoder/` |
| GloVe 6B 300d | [Stanford NLP](https://nlp.stanford.edu/projects/glove/) | `external_models/glove.6B.300d.txt` |

Also download [BlenderBot-small-90M](https://huggingface.co/facebook/blenderbot_small-90M) into `Blenderbot_small-90M/`.

### Data Preprocessing

```bash
cd _reformat && python process.py && cd ..
bash RUN/prepare_pmpc.sh
```

## Training

```bash
bash RUN/train_pmpc.sh
```

## Inference

```bash
bash RUN/infer_pmpc.sh
```

Select the best checkpoint from `DATA/pmpc.pmpc/` based on validation PPL, and set `--load_checkpoint` accordingly in the inference script.
