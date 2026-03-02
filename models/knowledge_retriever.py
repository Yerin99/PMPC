# coding=utf-8
"""
Knowledge Retriever for PMPC using DPR

논문 Section III-A-2: Prior Knowledge Cues
1. DPR로 training set에서 유사한 응답 검색
2. Top-l 응답을 encoder로 임베딩
3. He (prior knowledge cue) 생성

Reference:
- Eq. 3: similarity(H, r) = Ec(H)^T Eq(r)
- Eq. 4: He = Enc(r1 ⊕ ... ⊕ rl)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from typing import List, Dict, Optional, Tuple
import json
import logging
import os

logger = logging.getLogger(__name__)


class KnowledgeRetriever(nn.Module):
    """
    Prior Knowledge Cues Retriever using DPR
    
    논문 Section III-A-2 구현:
    1. DPR context/question encoder로 similarity 계산
    2. Training set에서 top-l 응답 검색
    3. Encoder로 임베딩하여 He 생성
    """
    
    def __init__(
        self,
        ctx_encoder_path: str = './external_models/dpr-ctx-encoder',
        question_encoder_path: str = './external_models/dpr-question-encoder',
        hidden_dim: int = 512,
        top_l: int = 5,
        device: str = 'cuda',
        training_responses_path: str = './_reformat/train.txt',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_l = top_l
        self._init_device = device
        
        # Load DPR encoders
        logger.info(f"Loading DPR context encoder from {ctx_encoder_path}...")
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_encoder_path)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(ctx_encoder_path)
        
        logger.info(f"Loading DPR question encoder from {question_encoder_path}...")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_path)
        self.q_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_path)
        
        # Freeze DPR parameters
        for param in self.ctx_encoder.parameters():
            param.requires_grad = False
        for param in self.q_encoder.parameters():
            param.requires_grad = False
        
        self.ctx_encoder.eval()
        self.q_encoder.eval()
        
        # DPR hidden dim (768 for BERT-base)
        self.dpr_hidden_dim = self.ctx_encoder.config.hidden_size
        
        # Projection: DPR dim -> hidden_dim
        self.projection = nn.Linear(self.dpr_hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)

        # Eq. 7: he = 1/l * Σ h_e^i — simple average (no learnable aggregator)

        # Load training responses and precompute embeddings
        self.training_responses = []
        self.response_embeddings = None
        self._load_training_responses(training_responses_path)
        
        logger.info(f"KnowledgeRetriever initialized: top_l={top_l}, dpr_dim={self.dpr_hidden_dim}")

    @property
    def device(self):
        """파라미터에서 실제 device를 가져옴 (model.to() 자동 추적)"""
        return self.projection.weight.device

    def _load_training_responses(self, path: str):
        """Load and encode training responses"""
        if not os.path.exists(path):
            logger.warning(f"Training responses file not found: {path}")
            return
        
        logger.info(f"Loading training responses from {path}...")
        
        # Load responses from training data (ESConv format)
        responses = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # ESConv format: dialog list with 'text' and 'speaker'
                    if 'dialog' in data:
                        for turn in data['dialog']:
                            if turn.get('speaker') == 'sys' and 'text' in turn:
                                responses.append(turn['text'])
                    # Also try simple 'response' field
                    elif 'response' in data:
                        responses.append(data['response'])
                except Exception as e:
                    continue
        
        # Deduplicate and limit
        self.training_responses = list(set(responses))[:10000]  # Limit for memory
        logger.info(f"Loaded {len(self.training_responses)} unique training responses")
    
    @torch.no_grad()
    def _encode_contexts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts using context encoder"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.ctx_tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            outputs = self.ctx_encoder(**inputs)
            embeddings = outputs.pooler_output  # (batch, 768)
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    @torch.no_grad()
    def _encode_queries(self, texts: List[str]) -> torch.Tensor:
        """Encode queries using question encoder"""
        inputs = self.q_tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        outputs = self.q_encoder(**inputs)
        return outputs.pooler_output  # (batch, 768)
    
    def precompute_response_embeddings(self):
        """Precompute embeddings for all training responses"""
        if not self.training_responses:
            logger.warning("No training responses to encode")
            return
        
        logger.info("Precomputing response embeddings...")
        self.response_embeddings = self._encode_contexts(self.training_responses)
        logger.info(f"Precomputed embeddings shape: {self.response_embeddings.shape}")
    
    @torch.no_grad()
    def retrieve(
        self, 
        dialogue_histories: List[str]
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Retrieve top-l similar responses from training set
        
        논문 Eq. 3: similarity(H, r) = Ec(H)^T Eq(r)
        
        Args:
            dialogue_histories: List of dialogue contexts
        
        Returns:
            retrieved_responses: List of List of top-l responses
            scores: (batch, top_l) similarity scores
        """
        if self.response_embeddings is None:
            self.precompute_response_embeddings()
        
        if self.response_embeddings is None or len(self.training_responses) == 0:
            # Fallback: return empty
            batch_size = len(dialogue_histories)
            return [[''] * self.top_l] * batch_size, torch.zeros(batch_size, self.top_l)
        
        # Encode queries
        query_embeddings = self._encode_queries(dialogue_histories)  # (batch, 768)
        
        # Compute similarities
        # similarity = query @ response.T
        similarities = torch.matmul(
            query_embeddings, 
            self.response_embeddings.T
        )  # (batch, num_responses)
        
        # Get top-l
        top_scores, top_indices = torch.topk(similarities, k=min(self.top_l, len(self.training_responses)), dim=-1)
        
        # Get responses
        retrieved_responses = []
        for indices in top_indices.tolist():
            responses = [self.training_responses[i] for i in indices]
            # Pad if needed
            while len(responses) < self.top_l:
                responses.append('')
            retrieved_responses.append(responses)
        
        return retrieved_responses, top_scores
    
    def forward(
        self, 
        dialogue_histories: List[str]
    ) -> torch.Tensor:
        """
        Extract prior knowledge cues
        
        논문 Eq. 4: He = Enc(r1 ⊕ ... ⊕ rl)
        
        Args:
            dialogue_histories: List of dialogue contexts
        
        Returns:
            he: (batch, hidden_dim) - Prior knowledge cue embeddings
        """
        batch_size = len(dialogue_histories)
        
        try:
            # Retrieve similar responses
            retrieved_responses, _ = self.retrieve(dialogue_histories)
            
            # Encode retrieved responses
            all_response_embeddings = []
            for responses in retrieved_responses:
                # Encode each response
                response_embs = []
                for resp in responses[:self.top_l]:
                    if resp:
                        inputs = self.ctx_tokenizer(
                            resp,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=128
                        ).to(self.device)
                        with torch.no_grad():
                            outputs = self.ctx_encoder(**inputs)
                        emb = outputs.pooler_output.squeeze(0)  # (768,)
                    else:
                        emb = torch.zeros(self.dpr_hidden_dim, device=self.device)
                    response_embs.append(emb)
                
                # Pad if needed
                while len(response_embs) < self.top_l:
                    response_embs.append(torch.zeros(self.dpr_hidden_dim, device=self.device))
                
                # Stack: (top_l, 768)
                response_embs = torch.stack(response_embs)
                all_response_embeddings.append(response_embs)
            
            # Stack: (batch, top_l, 768)
            all_response_embeddings = torch.stack(all_response_embeddings)
            
            # Project to hidden_dim: (batch, top_l, hidden_dim)
            projected = self.projection(all_response_embeddings)

            # Eq. 7: he = 1/l * Σ h_e^i — simple average
            he = projected.mean(dim=1)  # (batch, hidden_dim)
            
            # NaN 체크
            if torch.isnan(he).any() or torch.isinf(he).any():
                logger.warning("NaN/Inf in knowledge retriever output, returning zeros")
                return torch.zeros(batch_size, self.hidden_dim, device=self.device)
            
            return he
            
        except Exception as e:
            logger.warning(f"Knowledge retriever error: {e}, returning zeros")
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)
