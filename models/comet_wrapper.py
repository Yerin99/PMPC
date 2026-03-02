# coding=utf-8
"""
COMET-ATOMIC 2020 Wrapper for PMPC

논문 Section III-A-3: Mental State Cues
COMET을 사용하여 9개 relation 추출:
- User (6): xAttr, xEffect, xIntent, xNeed, xReact, xWant
- Listener (3): oEffect, oReact, oWant

Reference: https://github.com/allenai/comet-atomic-2020
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# COMET relation types
USER_RELATIONS = ['xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']
LISTENER_RELATIONS = ['oEffect', 'oReact', 'oWant']
ALL_RELATIONS = USER_RELATIONS + LISTENER_RELATIONS


class COMETWrapper(nn.Module):
    """
    COMET-ATOMIC 2020 Wrapper for Mental State Cues extraction
    
    논문 Eq. 5: Hc_i = Enc(COMET(post, relation_i))
    """
    
    def __init__(
        self,
        model_path: str = './external_models/comet-atomic-2020',
        hidden_dim: int = 512,
        device: str = 'cuda',
        max_length: int = 64,
        num_beams: int = 3,
        num_return_sequences: int = 1,
    ):
        super().__init__()
        self._init_device = device
        self.max_length = max_length
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.hidden_dim = hidden_dim
        
        # Load COMET model
        logger.info(f"Loading COMET-ATOMIC 2020 from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval()
        
        # Freeze COMET parameters (논문에서는 freeze)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # BART hidden dim (typically 1024 for BART-large)
        self.comet_hidden_dim = self.model.config.d_model
        
        # Projection to target hidden_dim (논문 Eq. 5의 Enc 역할)
        self.projection = nn.Linear(self.comet_hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)
        
        logger.info(f"COMET loaded: comet_dim={self.comet_hidden_dim}, target_dim={hidden_dim}")

    @property
    def device(self):
        """파라미터에서 실제 device를 가져옴 (model.to() 자동 추적)"""
        return self.projection.weight.device

    def _format_input(self, text: str, relation: str) -> str:
        """Format input for COMET: "{text} {relation} [GEN]" """
        return f"{text} {relation} [GEN]"
    
    @torch.no_grad()
    def generate_inference(
        self, 
        texts: List[str], 
        relation: str
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate commonsense inference for a batch of texts
        
        Args:
            texts: List of input texts (user utterances)
            relation: COMET relation type
        
        Returns:
            generated_texts: List of generated commonsense inferences
            hidden_states: (batch, hidden_dim) - mean pooled encoder hidden states
        """
        # Format inputs
        inputs = [self._format_input(text, relation) for text in texts]
        
        # Tokenize
        encoded = self.tokenizer(
            inputs, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Get encoder hidden states (for embedding)
        encoder_outputs = self.model.get_encoder()(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            return_dict=True
        )
        
        # Mean pooling of encoder hidden states
        mask = encoded['attention_mask'].unsqueeze(-1).float()
        hidden_states = (encoder_outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        # Project to target dimension
        projected = self.projection(hidden_states)  # (batch, hidden_dim)
        
        # Generate text (optional, for debugging)
        # generated_ids = self.model.generate(
        #     input_ids=encoded['input_ids'],
        #     attention_mask=encoded['attention_mask'],
        #     max_length=self.max_length,
        #     num_beams=self.num_beams,
        #     num_return_sequences=self.num_return_sequences,
        # )
        # generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        generated_texts = [""] * len(texts)  # Placeholder
        
        return generated_texts, projected
    
    def extract_mental_states(
        self, 
        user_utterances: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all mental state cues using COMET
        
        논문 Eq. 5: 9개 relation에 대한 commonsense 추론
        
        Args:
            user_utterances: List of user's last utterances (batch)
        
        Returns:
            Dict containing:
                - user_mental: (batch, 6, hidden_dim) - Hc1~Hc6
                - listener_mental: (batch, 3, hidden_dim) - Hc7~Hc9
        """
        batch_size = len(user_utterances)
        
        # User mental states (6 relations)
        user_mental_list = []
        for relation in USER_RELATIONS:
            _, hidden = self.generate_inference(user_utterances, relation)
            user_mental_list.append(hidden)
        user_mental = torch.stack(user_mental_list, dim=1)  # (batch, 6, hidden)
        
        # Listener mental states (3 relations)
        listener_mental_list = []
        for relation in LISTENER_RELATIONS:
            _, hidden = self.generate_inference(user_utterances, relation)
            listener_mental_list.append(hidden)
        listener_mental = torch.stack(listener_mental_list, dim=1)  # (batch, 3, hidden)
        
        return {
            'user_mental': user_mental,
            'listener_mental': listener_mental,
        }
    
    def forward(
        self, 
        user_utterances: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - same as extract_mental_states"""
        return self.extract_mental_states(user_utterances)


class COMETBatchWrapper(COMETWrapper):
    """
    Batch-optimized COMET wrapper for faster training
    
    모든 relation을 한 번에 처리하여 효율성 향상
    """
    
    @torch.no_grad()
    def extract_mental_states_batch(
        self, 
        user_utterances: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Batch-optimized mental state extraction
        
        모든 relation을 한 번에 처리 (9 * batch_size inputs)
        """
        batch_size = len(user_utterances)
        
        # Prepare all inputs at once
        all_inputs = []
        for text in user_utterances:
            for relation in ALL_RELATIONS:
                all_inputs.append(self._format_input(text, relation))
        
        # Tokenize all at once
        encoded = self.tokenizer(
            all_inputs, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Get encoder hidden states
        encoder_outputs = self.model.get_encoder()(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            return_dict=True
        )
        
        # Mean pooling
        mask = encoded['attention_mask'].unsqueeze(-1).float()
        hidden_states = (encoder_outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        # Project to target dimension
        projected = self.projection(hidden_states)  # (batch*9, hidden_dim)
        
        # Reshape: (batch*9, hidden) -> (batch, 9, hidden)
        projected = projected.view(batch_size, len(ALL_RELATIONS), self.hidden_dim)
        
        # Split into user (6) and listener (3)
        user_mental = projected[:, :6, :]  # (batch, 6, hidden)
        listener_mental = projected[:, 6:, :]  # (batch, 3, hidden)
        
        return {
            'user_mental': user_mental,
            'listener_mental': listener_mental,
        }
    
    def forward(self, user_utterances: List[str]) -> Dict[str, torch.Tensor]:
        return self.extract_mental_states_batch(user_utterances)
