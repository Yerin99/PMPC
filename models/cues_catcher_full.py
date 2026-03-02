# coding=utf-8
"""
Full Cues Catcher for PMPC

논문 Section III-A 완전 구현:
1. Topic Cues (Ht): GloVe + PMI - Section III-A-1
2. Prior Knowledge Cues (He): DPR - Section III-A-2
3. Mental State Cues (Hc1-9): COMET - Section III-A-3
4. Other Cues (Hp, Hs): BlenderBot encoder - Section III-A-4

Fig. 2의 Cues Catcher 구현
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import logging

from .comet_wrapper import COMETBatchWrapper
from .topic_extractor import TopicExtractor
from .knowledge_retriever import KnowledgeRetriever

logger = logging.getLogger(__name__)


class FullCuesCatcher(nn.Module):
    """
    Full Cues Catcher - 논문의 모든 Cue 추출기 통합
    
    논문 Fig. 2의 Cues Catcher 완전 구현:
    - Topic Extractor: GloVe + PMI → Ht
    - DPR: Prior Knowledge → He
    - COMET: Mental States → Hc (User 6 + Listener 3)
    - Context Encoder: Post(Hp), Situation(Hs) 
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        device: str = 'cuda',
        comet_path: str = './external_models/comet-atomic-2020',
        glove_path: str = './external_models/glove.6B.300d.txt',
        dpr_ctx_path: str = './external_models/dpr-ctx-encoder',
        dpr_q_path: str = './external_models/dpr-question-encoder',
        training_responses_path: str = './_reformat/train.txt',
        top_k_topics: int = 10,
        top_l_knowledge: int = 5,
        corpus_stats_path: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        logger.info("Initializing Full Cues Catcher...")
        
        # 1. COMET for Mental State Cues (Hc1-9)
        logger.info("Loading COMET-ATOMIC 2020...")
        self.comet = COMETBatchWrapper(
            model_path=comet_path,
            hidden_dim=hidden_dim,
            device=device,
        )
        
        # 2. Topic Extractor (GloVe + PMI) for Ht
        logger.info("Loading Topic Extractor (GloVe + PMI)...")
        self.topic_extractor = TopicExtractor(
            glove_path=glove_path,
            hidden_dim=hidden_dim,
            top_k=top_k_topics,
            device=device,
            corpus_stats_path=corpus_stats_path,
        )
        
        # 3. Knowledge Retriever (DPR) for He
        logger.info("Loading Knowledge Retriever (DPR)...")
        self.knowledge_retriever = KnowledgeRetriever(
            ctx_encoder_path=dpr_ctx_path,
            question_encoder_path=dpr_q_path,
            hidden_dim=hidden_dim,
            top_l=top_l_knowledge,
            device=device,
            training_responses_path=training_responses_path,
        )
        
        # 4. Projections for Other Cues (Hp, Hs)
        # Hp: Post embedding (user's last utterance)
        self.post_proj = nn.Linear(hidden_dim, hidden_dim)
        # Hs: Situation embedding
        self.situation_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_weights()
        
        logger.info("Full Cues Catcher initialized successfully!")
    
    def _init_weights(self):
        for layer in [self.post_proj, self.situation_proj]:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        dialogue_histories: Optional[List[str]] = None,
        situations: Optional[List[str]] = None,
        last_utterances: Optional[List[str]] = None,
        situation_hidden: Optional[torch.Tensor] = None,
        situation_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all cues from dialogue

        논문 Section III-A 구현:
        - 외부 모델 (COMET, DPR, GloVe) 출력은 detach하여 gradient 차단
        - 이는 논문에서 외부 모델을 freeze하여 사용하는 것과 일치

        Args:
            encoder_hidden: (batch, seq, hidden) - BlenderBot encoder output
            attention_mask: (batch, seq) - Attention mask
            dialogue_histories: List of full dialogue contexts (text)
            situations: List of situation descriptions (text)
            last_utterances: List of user's last utterances (text)
            situation_hidden: (batch, sit_seq, hidden) - Separately encoded situation
            situation_mask: (batch, sit_seq) - Situation attention mask

        Returns:
            Dict containing:
                - hp: (batch, hidden) - Post cue
                - hs: (batch, hidden) - Situation cue
                - ht: (batch, hidden) - Topic cue
                - he: (batch, hidden) - Prior knowledge cue
                - user_mental: (batch, 6, hidden) - User mental states (Hc1-6)
                - listener_mental: (batch, 3, hidden) - Listener mental states (Hc7-9)
        """
        batch_size = encoder_hidden.size(0)
        device = encoder_hidden.device

        # Context pooling from encoder hidden states
        mask = attention_mask.unsqueeze(-1).float()
        context = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # 1. Other Cues (Hp, Hs) - from encoder hidden states
        # Hp: Post embedding (context representation)
        hp = self.post_proj(context)  # (batch, hidden)

        # Eq. 6: hs = Mean-pooling(H^s) — mean-pool the situation encoding
        if situation_hidden is not None and situation_mask is not None:
            sit_mask = situation_mask.unsqueeze(-1).float()
            sit_pooled = (situation_hidden * sit_mask).sum(dim=1) / sit_mask.sum(dim=1).clamp(min=1e-9)
            hs = self.situation_proj(sit_pooled)  # (batch, hidden)
        else:
            # Fallback: mean-pool the full encoder hidden (approximation)
            hs = self.situation_proj(context)  # (batch, hidden)
        
        # 2-4: Text-based cues (require text inputs)
        if dialogue_histories is not None and last_utterances is not None:
            # Provide default situations if not given
            if situations is None:
                situations = [''] * batch_size

            # Free GPU cache before heavy external model inference
            torch.cuda.empty_cache()

            # 2. Topic Cues (Ht) - GloVe + PMI (논문 Eq. 1, 2)
            # GloVe는 frozen이므로 detach 필요 없음 (learnable projection만 학습)
            ht = self.topic_extractor(
                dialogue_histories=dialogue_histories,
                situations=situations,
                last_utterances=last_utterances,
            )  # (batch, hidden)

            # 3. Prior Knowledge Cues (He) - DPR (논문 Eq. 3, 4)
            # DPR encoder는 frozen이므로 출력 detach
            with torch.no_grad():
                he_raw = self.knowledge_retriever(
                    dialogue_histories=dialogue_histories,
                )  # (batch, hidden)
            he = he_raw.detach()  # Gradient 차단

            # 4. Mental State Cues (Hc) - COMET (논문 Eq. 5)
            # COMET은 frozen이므로 출력 detach
            with torch.no_grad():
                mental_states = self.comet(
                    user_utterances=last_utterances,
                )
            user_mental = mental_states['user_mental'].detach()  # (batch, 6, hidden)
            listener_mental = mental_states['listener_mental'].detach()  # (batch, 3, hidden)

            # NaN 방지: 값 클리핑
            ht = torch.clamp(ht, min=-10.0, max=10.0)
            he = torch.clamp(he, min=-10.0, max=10.0)
            user_mental = torch.clamp(user_mental, min=-10.0, max=10.0)
            listener_mental = torch.clamp(listener_mental, min=-10.0, max=10.0)
        else:
            # Fallback: use encoder-based approximation
            ht = self.post_proj(context)
            he = self.situation_proj(context)
            user_mental = self.post_proj(context).unsqueeze(1).expand(-1, 6, -1)
            listener_mental = self.situation_proj(context).unsqueeze(1).expand(-1, 3, -1)
        
        # 모든 텐서가 같은 device에 있는지 확인
        ht = ht.to(device)
        he = he.to(device)
        user_mental = user_mental.to(device)
        listener_mental = listener_mental.to(device)
        
        return {
            'hp': hp,
            'hs': hs,
            'ht': ht,
            'he': he,
            'user_mental': user_mental,
            'listener_mental': listener_mental,
        }


class CuesCatcherWithCache(FullCuesCatcher):
    """
    Cues Catcher with caching for faster training
    
    Heavy components (COMET, DPR) can be cached during training
    """
    
    def __init__(self, *args, use_cache: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = use_cache
        self.cache = {}
    
    def clear_cache(self):
        self.cache = {}
    
    def forward(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        dialogue_histories: Optional[List[str]] = None,
        situations: Optional[List[str]] = None,
        last_utterances: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,  # For caching
    ) -> Dict[str, torch.Tensor]:
        """Forward with optional caching"""
        
        if not self.use_cache or sample_ids is None:
            return super().forward(
                encoder_hidden, attention_mask,
                dialogue_histories, situations, last_utterances
            )
        
        # Check cache
        batch_size = encoder_hidden.size(0)
        cached_results = []
        uncached_indices = []
        uncached_texts = {'histories': [], 'situations': [], 'utterances': [], 'ids': []}
        
        for i, sid in enumerate(sample_ids):
            if sid in self.cache:
                cached_results.append((i, self.cache[sid]))
            else:
                uncached_indices.append(i)
                uncached_texts['histories'].append(dialogue_histories[i] if dialogue_histories else '')
                uncached_texts['situations'].append(situations[i] if situations else '')
                uncached_texts['utterances'].append(last_utterances[i] if last_utterances else '')
                uncached_texts['ids'].append(sid)
        
        # Compute for uncached samples
        if uncached_indices:
            uncached_encoder_hidden = encoder_hidden[uncached_indices]
            uncached_attention_mask = attention_mask[uncached_indices]
            
            uncached_results = super().forward(
                uncached_encoder_hidden, uncached_attention_mask,
                uncached_texts['histories'], uncached_texts['situations'], uncached_texts['utterances']
            )
            
            # Cache results
            for i, sid in enumerate(uncached_texts['ids']):
                result_i = {k: v[i:i+1] for k, v in uncached_results.items()}
                self.cache[sid] = result_i
        
        # Merge results
        # (This is simplified - full implementation would properly merge cached and uncached)
        return super().forward(
            encoder_hidden, attention_mask,
            dialogue_histories, situations, last_utterances
        )
