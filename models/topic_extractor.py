# coding=utf-8
"""
Topic Extractor for PMPC

논문 Section III-A-1: Topic Cues
1. Part-of-speech tagging으로 명사구 추출
2. PMI로 user's last utterance와의 관련성 계산
3. Top-k 키워드 선택
4. GloVe embedding으로 Ht 생성

Reference:
- Eq. 1: PMI(w, p) = Σ log(p(w,wi) / p(w)p(wi))
- Eq. 2: Ht = {emb(w1), ..., emb(wk)}
"""

import torch
import torch.nn as nn
import numpy as np
import nltk
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
import logging
import os
import re
import json
import math

logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)


class GloVeEmbeddings:
    """GloVe Embeddings loader"""
    
    def __init__(self, glove_path: str, dim: int = 300):
        self.dim = dim
        self.word2idx = {}
        self.embeddings = None
        self._load_glove(glove_path)
    
    def _load_glove(self, glove_path: str):
        """Load GloVe embeddings from file"""
        logger.info(f"Loading GloVe embeddings from {glove_path}...")
        
        words = []
        vectors = []
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                values = line.strip().split()
                word = values[0]
                try:
                    vector = np.array(values[1:], dtype=np.float32)
                    if len(vector) == self.dim:
                        self.word2idx[word] = len(words)
                        words.append(word)
                        vectors.append(vector)
                except:
                    continue
                
                if i > 0 and i % 100000 == 0:
                    logger.info(f"Loaded {i} words...")
        
        self.embeddings = np.stack(vectors)
        logger.info(f"Loaded {len(words)} GloVe embeddings with dim={self.dim}")
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a word"""
        word = word.lower()
        if word in self.word2idx:
            return self.embeddings[self.word2idx[word]]
        return None
    
    def get_embeddings_batch(self, words: List[str]) -> Tuple[np.ndarray, List[bool]]:
        """Get embeddings for a batch of words"""
        embeddings = []
        found = []
        for word in words:
            emb = self.get_embedding(word)
            if emb is not None:
                embeddings.append(emb)
                found.append(True)
            else:
                embeddings.append(np.zeros(self.dim, dtype=np.float32))
                found.append(False)
        return np.stack(embeddings), found


class TopicExtractor(nn.Module):
    """
    Topic Cues Extractor using GloVe + PMI
    
    논문 Section III-A-1 구현:
    1. POS tagging으로 명사구 추출 (NN*, JJ)
    2. PMI로 relevance 계산
    3. Top-k 선택
    4. GloVe embedding
    """
    
    def __init__(
        self,
        glove_path: str = './external_models/glove.6B.300d.txt',
        hidden_dim: int = 512,
        glove_dim: int = 300,
        top_k: int = 10,
        device: str = 'cuda',
        corpus_stats_path: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.glove_dim = glove_dim
        self.top_k = top_k
        self._init_device = device

        # Load GloVe
        self.glove = GloVeEmbeddings(glove_path, dim=glove_dim)

        # Projection: GloVe dim -> hidden_dim
        self.projection = nn.Linear(glove_dim, hidden_dim)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)

        # Eq. 8: ht = 1/k * Σ H_i^t — simple average (no learnable aggregator)

        # Corpus statistics for proper PMI (Eq. 1)
        self.word_freq: Counter = Counter()
        self.cooccur_freq: Counter = Counter()
        self.total_docs: int = 0
        if corpus_stats_path and os.path.exists(corpus_stats_path):
            self._load_corpus_statistics(corpus_stats_path)

        logger.info(f"TopicExtractor initialized: top_k={top_k}, glove_dim={glove_dim}, hidden_dim={hidden_dim}")

    @property
    def device(self):
        """파라미터에서 실제 device를 가져옴 (model.to() 자동 추적)"""
        return self.projection.weight.device

    def build_corpus_statistics(self, corpus_texts: List[str], save_path: Optional[str] = None):
        """
        Pre-compute word frequency and co-occurrence statistics from the training corpus.

        Co-occurrence is defined at the document (utterance) level: two words
        co-occur if they appear in the same document.

        Args:
            corpus_texts: List of all training utterances / dialogue texts.
            save_path: If given, persist the statistics as JSON for fast reload.
        """
        logger.info("Building corpus statistics for PMI computation...")
        self.word_freq = Counter()
        self.cooccur_freq = Counter()
        self.total_docs = len(corpus_texts)

        for text in corpus_texts:
            tokens = set(nltk.word_tokenize(text.lower()))
            tokens = {t for t in tokens if t.isalpha()}
            for t in tokens:
                self.word_freq[t] += 1
            sorted_tokens = sorted(tokens)
            for i, t1 in enumerate(sorted_tokens):
                for t2 in sorted_tokens[i + 1:]:
                    self.cooccur_freq[(t1, t2)] += 1

        logger.info(
            f"Corpus stats built: {len(self.word_freq)} unique words, "
            f"{len(self.cooccur_freq)} co-occurrence pairs, {self.total_docs} docs"
        )

        if save_path:
            stats = {
                'word_freq': dict(self.word_freq),
                'cooccur_freq': {f"{k[0]}|||{k[1]}": v for k, v in self.cooccur_freq.items()},
                'total_docs': self.total_docs,
            }
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f)
            logger.info(f"Corpus statistics saved to {save_path}")

    def _load_corpus_statistics(self, path: str):
        """Load pre-computed corpus statistics from JSON."""
        logger.info(f"Loading corpus statistics from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        self.word_freq = Counter(stats['word_freq'])
        self.cooccur_freq = Counter()
        for k, v in stats['cooccur_freq'].items():
            parts = k.split('|||')
            if len(parts) == 2:
                self.cooccur_freq[(parts[0], parts[1])] = v
        self.total_docs = stats['total_docs']
        logger.info(f"Loaded corpus stats: {len(self.word_freq)} words, {self.total_docs} docs")

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases using POS tagging
        
        논문 Section III-A-1:
        "noun phrases are selected as candidate words by applying the regular 
        expression <NN.*|JJ>*<NN.*> after performing part-of-speech tagging"
        
        Pattern: <NN.*|JJ>*<NN.*>
        - Zero or more (NN* or JJ) followed by one or more NN*
        - 예: "good friend", "best friend ever", "job", "mental health"
        
        Note: 논문에서는 Stanford CoreNLP를 사용했지만, NLTK POS tagger로 대체
        (동일한 Penn Treebank tagset 사용)
        """
        try:
            tokens = nltk.word_tokenize(text.lower())
            pos_tags = nltk.pos_tag(tokens)
        except Exception as e:
            logger.debug(f"POS tagging failed: {e}")
            return []
        
        # Pattern: <NN.*|JJ>*<NN.*>
        # 0개 이상의 (NN* 또는 JJ) + 1개 이상의 NN*
        candidates = []
        i = 0
        while i < len(pos_tags):
            word, tag = pos_tags[i]
            
            # Check if current position can start a noun phrase
            # (either JJ, NN, NNS, NNP, NNPS)
            if tag.startswith('NN') or tag.startswith('JJ'):
                phrase = []
                j = i
                
                # Collect <NN.*|JJ>* part
                while j < len(pos_tags):
                    curr_word, curr_tag = pos_tags[j]
                    if curr_tag.startswith('NN') or curr_tag.startswith('JJ'):
                        phrase.append(curr_word)
                        j += 1
                    else:
                        break
                
                # Check if phrase ends with NN.* (required by pattern)
                if phrase:
                    # Find last NN in phrase
                    last_nn_idx = -1
                    for k in range(len(phrase) - 1, -1, -1):
                        # Check if original tag was NN*
                        orig_idx = i + k
                        if orig_idx < len(pos_tags) and pos_tags[orig_idx][1].startswith('NN'):
                            last_nn_idx = k
                            break
                    
                    if last_nn_idx >= 0:
                        # Trim phrase to end at last NN
                        valid_phrase = phrase[:last_nn_idx + 1]
                        if valid_phrase:
                            candidates.append(' '.join(valid_phrase))
                
                i = j if j > i else i + 1
            else:
                i += 1
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen and len(c) > 1:  # Skip single characters
                seen.add(c)
                unique_candidates.append(c)
        
        return unique_candidates
    
    def _compute_pmi(
        self,
        candidates: List[str],
        last_utterance: str,
        corpus_freq: Optional[Dict[str, int]] = None
    ) -> List[Tuple[str, float]]:
        """
        Compute PMI between candidates and the user's last utterance.

        논문 Eq. 1: PMI(w, p) = Σ_{wi ∈ p} log( p(w, wi) / (p(w) * p(wi)) )

        where p(w) = freq(w) / N, p(w, wi) = cooccur(w, wi) / N,
        and N is the total number of documents in the corpus.

        Falls back to overlap-based heuristic when corpus statistics are
        unavailable (i.e. build_corpus_statistics() was never called).
        """
        last_tokens = set(nltk.word_tokenize(last_utterance.lower()))
        last_tokens = {t for t in last_tokens if t.isalpha()}

        use_proper_pmi = self.total_docs > 0 and len(self.word_freq) > 0

        scored_candidates = []
        for candidate in candidates:
            candidate_tokens = set(candidate.lower().split())

            if use_proper_pmi:
                # Proper PMI (Eq. 1)
                pmi_score = 0.0
                for w in candidate_tokens:
                    pw = self.word_freq.get(w, 0) / self.total_docs
                    if pw == 0:
                        continue
                    for wi in last_tokens:
                        pwi = self.word_freq.get(wi, 0) / self.total_docs
                        if pwi == 0:
                            continue
                        pair = tuple(sorted([w, wi]))
                        pwwi = self.cooccur_freq.get(pair, 0) / self.total_docs
                        if pwwi > 0:
                            pmi_score += math.log(pwwi / (pw * pwi))
                scored_candidates.append((candidate, pmi_score))
            else:
                # Fallback: overlap-based heuristic
                overlap = len(candidate_tokens & last_tokens)
                if overlap > 0:
                    pmi_score = np.log(overlap + 1) / (np.log(len(candidate_tokens) + 1) + 1e-9)
                else:
                    pmi_score = 0.0
                scored_candidates.append((candidate, pmi_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates
    
    def extract_topics(
        self, 
        dialogue_history: str, 
        situation: str,
        last_utterance: str
    ) -> torch.Tensor:
        """
        Extract topic cues from dialogue
        
        Args:
            dialogue_history: Full dialogue context
            situation: Situation description
            last_utterance: User's last utterance
        
        Returns:
            ht: (hidden_dim,) - Topic cue embedding
        """
        # Combine text sources
        full_text = f"{situation} {dialogue_history}"
        
        # Extract noun phrases
        candidates = self._extract_noun_phrases(full_text)
        
        if not candidates:
            # Fallback: use simple tokenization
            tokens = nltk.word_tokenize(full_text.lower())
            candidates = [t for t in tokens if len(t) > 2 and t.isalpha()]
        
        # Compute PMI and get top-k
        scored = self._compute_pmi(candidates, last_utterance)
        top_candidates = [w for w, _ in scored[:self.top_k]]
        
        # Pad if needed
        while len(top_candidates) < self.top_k:
            top_candidates.append('')
        
        # Get GloVe embeddings
        embeddings = []
        for word in top_candidates[:self.top_k]:
            if word:
                # For multi-word phrases, average word embeddings
                word_tokens = word.split()
                word_embs = []
                for token in word_tokens:
                    emb = self.glove.get_embedding(token)
                    if emb is not None:
                        word_embs.append(emb)
                if word_embs:
                    embeddings.append(np.mean(word_embs, axis=0))
                else:
                    embeddings.append(np.zeros(self.glove_dim, dtype=np.float32))
            else:
                embeddings.append(np.zeros(self.glove_dim, dtype=np.float32))
        
        # Convert to tensor
        embeddings = torch.tensor(np.stack(embeddings), dtype=torch.float32)  # (top_k, glove_dim)
        
        return embeddings
    
    def forward(
        self, 
        dialogue_histories: List[str], 
        situations: List[str],
        last_utterances: List[str]
    ) -> torch.Tensor:
        """
        Batch forward pass
        
        논문 Section III-A-1: Topic Cues 추출
        1. POS tagging으로 명사구 추출 (논문: Stanford CoreNLP, 여기서는 NLTK)
        2. PMI로 relevance 계산
        3. Top-k 선택
        4. GloVe embedding
        
        Args:
            dialogue_histories: List of dialogue histories
            situations: List of situation descriptions
            last_utterances: List of user's last utterances
        
        Returns:
            ht: (batch, hidden_dim) - Topic cue embeddings
        """
        batch_size = len(dialogue_histories)
        
        try:
            # Extract topics for each sample
            all_embeddings = []
            for i in range(batch_size):
                emb = self.extract_topics(
                    dialogue_histories[i], 
                    situations[i], 
                    last_utterances[i]
                )
                all_embeddings.append(emb)
            
            # Stack: (batch, top_k, glove_dim)
            all_embeddings = torch.stack(all_embeddings).to(self.device)
            
            # Project to hidden_dim: (batch, top_k, hidden_dim)
            projected = self.projection(all_embeddings)

            # Eq. 8: ht = 1/k * Σ H_i^t — simple average
            ht = projected.mean(dim=1)  # (batch, hidden_dim)
            
            # NaN 체크
            if torch.isnan(ht).any() or torch.isinf(ht).any():
                logger.warning("NaN/Inf in topic extractor output, returning zeros")
                return torch.zeros(batch_size, self.hidden_dim, device=self.device)
            
            return ht
            
        except Exception as e:
            logger.warning(f"Topic extractor error: {e}, returning zeros")
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)
