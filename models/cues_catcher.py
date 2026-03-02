# coding=utf-8
"""
PMPC Cues Catcher Module

논문 Section III-A 구현:
1) Topic Cues (Ht) - PMI 기반 키워드 추출 + GloVe embeddings
2) Prior Knowledge Cues (He) - DPR 기반 유사 응답 검색 
3) Mental State Cues - COMET 기반 user/listener 심리 상태
4) Other Cues - Situation (Hs), Post (Hp)

실용적 구현:
- GloVe, DPR, COMET 대신 BlenderBot encoder + 학습 가능한 embeddings 사용
- 논문의 핵심 아이디어(다각적 단서 추출)는 유지
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TopicCuesExtractor(nn.Module):
    """
    Topic Cues Extractor (논문 Eq. 1-2)
    
    논문: PMI로 top-k 키워드 선택 후 GloVe embedding
    구현: Encoder hidden states에서 attention으로 topic 추출
    """
    def __init__(self, hidden_dim: int, topic_dim: int = 300, top_k: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topic_dim = topic_dim
        self.top_k = top_k
        
        # Topic attention (PMI 대신 attention 사용)
        self.topic_attention = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.topic_attention.weight, gain=0.1)
        nn.init.zeros_(self.topic_attention.bias)
        
        # Topic projection (GloVe 대신 projection)
        self.topic_projection = nn.Linear(hidden_dim, topic_dim)
        nn.init.xavier_uniform_(self.topic_projection.weight, gain=0.1)
        nn.init.zeros_(self.topic_projection.bias)
        
    def forward(self, encoder_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_hidden: (batch, seq, hidden_dim)
            attention_mask: (batch, seq)
        
        Returns:
            topic_cues: (batch, topic_dim) - ht (논문 Eq. 8)
        """
        # Attention scores
        scores = self.topic_attention(encoder_hidden).squeeze(-1)  # (batch, seq)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)  # (batch, seq)
        
        # Weighted sum
        weighted = torch.bmm(weights.unsqueeze(1), encoder_hidden).squeeze(1)  # (batch, hidden)
        
        # Project to topic dimension
        topic_cues = self.topic_projection(weighted)  # (batch, topic_dim)
        
        return topic_cues


class PriorKnowledgeCuesExtractor(nn.Module):
    """
    Prior Knowledge Cues Extractor (논문 Eq. 3-4, 7)
    
    논문: DPR로 top-l 유사 응답 검색 후 Encoder로 인코딩
    구현: 학습 가능한 knowledge bank + attention
    """
    def __init__(self, hidden_dim: int, num_knowledge: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_knowledge = num_knowledge
        
        # Learnable knowledge bank (DPR 대신)
        self.knowledge_bank = nn.Parameter(torch.randn(num_knowledge, hidden_dim) * 0.01)
        
        # Context-aware retrieval
        self.retrieval_query = nn.Linear(hidden_dim, hidden_dim)
        self.retrieval_key = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.retrieval_query.weight, gain=0.1)
        nn.init.zeros_(self.retrieval_query.bias)
        nn.init.xavier_uniform_(self.retrieval_key.weight, gain=0.1)
        nn.init.zeros_(self.retrieval_key.bias)
        
    def forward(self, context_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_hidden: (batch, hidden_dim) - mean pooled encoder output
        
        Returns:
            knowledge_cues: (batch, hidden_dim) - he (논문 Eq. 7)
        """
        batch_size = context_hidden.size(0)
        
        # Query from context
        query = self.retrieval_query(context_hidden)  # (batch, hidden)
        
        # Keys from knowledge bank
        keys = self.retrieval_key(self.knowledge_bank)  # (num_knowledge, hidden)
        
        # Attention scores (similarity)
        scores = torch.matmul(query, keys.t())  # (batch, num_knowledge)
        weights = F.softmax(scores / (self.hidden_dim ** 0.5), dim=-1)  # (batch, num_knowledge)
        
        # Weighted sum of knowledge (논문 Eq. 7: average of top-l)
        knowledge_cues = torch.matmul(weights, self.knowledge_bank)  # (batch, hidden)
        
        return knowledge_cues


class MentalStateCuesExtractor(nn.Module):
    """
    Mental State Cues Extractor (논문 Eq. 5)
    
    논문: COMET으로 9개 relation 추출
    - User (6): xAttr, xEffect, xIntent, xNeed, xReact, xWant
    - Listener (3): oEffect, oReact, oWant
    
    구현: 학습 가능한 relation embeddings + context projection
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 9개 relation embeddings (COMET 대신)
        self.num_user_relations = 6  # xAttr, xEffect, xIntent, xNeed, xReact, xWant
        self.num_listener_relations = 3  # oEffect, oReact, oWant
        
        # Relation type embeddings (작은 값으로 초기화)
        self.user_relation_embeds = nn.Parameter(
            torch.randn(self.num_user_relations, hidden_dim) * 0.01
        )
        self.listener_relation_embeds = nn.Parameter(
            torch.randn(self.num_listener_relations, hidden_dim) * 0.01
        )
        
        # Context-conditioned projection
        self.user_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )
        self.listener_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )
        
        # Initialize projections
        for proj in [self.user_projection, self.listener_projection]:
            for layer in proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
        
    def forward(self, post_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            post_hidden: (batch, hidden_dim) - user's last utterance representation
        
        Returns:
            user_mental: (batch, 6, hidden_dim) - Hc1~Hc6
            listener_mental: (batch, 3, hidden_dim) - Hc7~Hc9
        """
        batch_size = post_hidden.size(0)
        
        # User mental states (Hc1 ~ Hc6)
        # 각 relation embedding과 post를 결합
        user_mental = []
        for i in range(self.num_user_relations):
            rel_embed = self.user_relation_embeds[i].unsqueeze(0).expand(batch_size, -1)  # (batch, hidden)
            combined = torch.cat([post_hidden, rel_embed], dim=-1)  # (batch, hidden*2)
            mental_state = self.user_projection(combined)  # (batch, hidden)
            user_mental.append(mental_state)
        user_mental = torch.stack(user_mental, dim=1)  # (batch, 6, hidden)
        
        # Listener mental states (Hc7 ~ Hc9)
        listener_mental = []
        for i in range(self.num_listener_relations):
            rel_embed = self.listener_relation_embeds[i].unsqueeze(0).expand(batch_size, -1)
            combined = torch.cat([post_hidden, rel_embed], dim=-1)
            mental_state = self.listener_projection(combined)
            listener_mental.append(mental_state)
        listener_mental = torch.stack(listener_mental, dim=1)  # (batch, 3, hidden)
        
        return user_mental, listener_mental


class CuesCatcher(nn.Module):
    """
    PMPC Cues Catcher (논문 Section III-A 전체)
    
    모든 cues를 추출하는 통합 모듈:
    - Topic Cues (ht)
    - Prior Knowledge Cues (he)
    - User Mental States (Hc1~Hc6)
    - Listener Mental States (Hc7~Hc9)
    - Situation Cue (hs)
    - Post Cue (hp)
    """
    def __init__(self, hidden_dim: int, topic_dim: int = 300, num_knowledge: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topic_dim = topic_dim
        
        # Sub-modules
        self.topic_extractor = TopicCuesExtractor(hidden_dim, topic_dim)
        self.knowledge_extractor = PriorKnowledgeCuesExtractor(hidden_dim, num_knowledge)
        self.mental_extractor = MentalStateCuesExtractor(hidden_dim)
        
        # Situation/Post projection
        self.situation_projection = nn.Linear(hidden_dim, hidden_dim)
        self.post_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize projections
        for layer in [self.situation_projection, self.post_projection]:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
        
        logger.info(f"CuesCatcher initialized: hidden_dim={hidden_dim}, topic_dim={topic_dim}")
    
    def forward(
        self, 
        encoder_hidden: torch.Tensor, 
        attention_mask: torch.Tensor,
        post_hidden: Optional[torch.Tensor] = None,
        situation_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        모든 cues 추출
        
        Args:
            encoder_hidden: (batch, seq, hidden_dim) - encoder output
            attention_mask: (batch, seq)
            post_hidden: (batch, hidden_dim) - user's last utterance (optional)
            situation_hidden: (batch, hidden_dim) - situation embedding (optional)
        
        Returns:
            Dict containing:
                - ht: topic cues (batch, topic_dim)
                - he: prior knowledge cues (batch, hidden_dim)
                - user_mental: (batch, 6, hidden_dim) - Hc1~Hc6
                - listener_mental: (batch, 3, hidden_dim) - Hc7~Hc9
                - hs: situation cue (batch, hidden_dim)
                - hp: post cue (batch, hidden_dim)
        """
        # Mean pooling for context
        mask = attention_mask.unsqueeze(-1).float()
        context_hidden = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        # Post hidden (default: context)
        if post_hidden is None:
            post_hidden = context_hidden
        
        # Situation hidden (default: first token)
        if situation_hidden is None:
            situation_hidden = encoder_hidden[:, 0, :]
        
        # Extract all cues
        ht = self.topic_extractor(encoder_hidden, attention_mask)  # (batch, topic_dim)
        he = self.knowledge_extractor(context_hidden)  # (batch, hidden_dim)
        user_mental, listener_mental = self.mental_extractor(post_hidden)
        
        # Situation and Post cues (논문 Eq. 6과 유사)
        hs = self.situation_projection(situation_hidden)  # (batch, hidden_dim)
        hp = self.post_projection(post_hidden)  # (batch, hidden_dim)
        
        return {
            'ht': ht,
            'he': he,
            'user_mental': user_mental,  # (batch, 6, hidden)
            'listener_mental': listener_mental,  # (batch, 3, hidden)
            'hs': hs,
            'hp': hp,
        }
