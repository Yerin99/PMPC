# coding=utf-8
"""
PMPC Model - 논문 완전 구현 버전 (Full Cues Catcher 포함)

핵심 구현:
1. Full Cues Catcher: COMET + GloVe + DPR (논문 Section III-A)
2. Prompt Builder: Pe, Pd, Strategy Codebook (논문 Section III-B)
3. Response Generator: Prepend 방식 (논문 Section III-C, Eq. 13, 16)

Reference: [IEEE 2025] PMPC - Prompt Learning with Multiperspective Cues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
try:
    from transformers.generation_utils import top_k_top_p_filtering
except ImportError:
    def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

from transformers.models.blenderbot_small import (
    BlenderbotSmallConfig, 
    BlenderbotSmallForConditionalGeneration,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from .PARAMS import SAMPLE, TEMPERATURE
import logging
import os

logger = logging.getLogger(__name__)

# Check if full cues catcher components are available
USE_FULL_CUES_CATCHER = True
try:
    from .cues_catcher_full import FullCuesCatcher
    logger.info("Full Cues Catcher (COMET + GloVe + DPR) available")
except ImportError as e:
    USE_FULL_CUES_CATCHER = False
    logger.warning(f"Full Cues Catcher not available: {e}. Using lightweight version.")


class PMPCPromptBuilder(nn.Module):
    """PMPC Prompt Builder - 논문 Section III-B"""
    
    def __init__(self, hidden_dim: int = 512, prompt_len: int = 20, n_strategies: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prompt_len = prompt_len
        self.n_strategies = n_strategies
        
        # Eq. 10: hx = Wx[Hc1 ⊕ ... ⊕ Hc6]
        self.Wx = nn.Linear(hidden_dim * 6, hidden_dim)
        # Eq. 12: ho = Wo[Hc7 ⊕ Hc8 ⊕ Hc9]
        self.Wo = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Eq. 9: Pe = MLPe(hp ⊕ hs ⊕ hx)
        self.MLPe = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, prompt_len * hidden_dim),
        )
        
        # Eq. 11: Pd = MLPd(hp ⊕ ht ⊕ he ⊕ ho ⊕ hstrat)
        self.MLPd = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, prompt_len * hidden_dim),
        )
        
        # Eq. 14: Strategy predictor
        self.strategy_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_strategies)
        )
        
        # Eq. 15: Strategy Codebook
        self.Estrat = nn.Embedding(n_strategies, hidden_dim)
        
        # Normalization
        self.pe_norm = nn.LayerNorm(hidden_dim)
        self.pd_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.Wx, self.Wo]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
        for mlp in [self.MLPe, self.MLPd, self.strategy_predictor]:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
        nn.init.normal_(self.Estrat.weight, mean=0.0, std=0.02)
    
    def compute_hx(self, user_mental: torch.Tensor) -> torch.Tensor:
        batch_size = user_mental.size(0)
        return self.Wx(user_mental.view(batch_size, -1))
    
    def compute_ho(self, listener_mental: torch.Tensor) -> torch.Tensor:
        batch_size = listener_mental.size(0)
        return self.Wo(listener_mental.view(batch_size, -1))
    
    def compute_Pe(self, hp, hs, hx) -> torch.Tensor:
        batch_size = hp.size(0)
        combined = torch.cat([hp, hs, hx], dim=-1)
        pe_flat = self.MLPe(combined)
        Pe = pe_flat.view(batch_size, self.prompt_len, self.hidden_dim)
        return self.pe_norm(Pe)
    
    def predict_strategy(self, Hcls):
        logits = self.strategy_predictor(Hcls)
        probs = F.softmax(logits, dim=-1)
        return logits, probs
    
    def compute_hstrat(self, strategy_probs):
        return torch.matmul(strategy_probs, self.Estrat.weight)
    
    def compute_Pd(self, hp, ht, he, ho, hstrat) -> torch.Tensor:
        batch_size = hp.size(0)
        combined = torch.cat([hp, ht, he, ho, hstrat], dim=-1)
        pd_flat = self.MLPd(combined)
        Pd = pd_flat.view(batch_size, self.prompt_len, self.hidden_dim)
        return self.pd_norm(Pd)


class LightweightCuesCatcher(nn.Module):
    """Lightweight Cues Catcher (Fallback when COMET/GloVe/DPR not available)"""
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.user_mental_proj = nn.Linear(hidden_dim, hidden_dim * 6)
        self.listener_mental_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.topic_proj = nn.Linear(hidden_dim, hidden_dim)
        self.knowledge_proj = nn.Linear(hidden_dim, hidden_dim)
        self.situation_proj = nn.Linear(hidden_dim, hidden_dim)
        self.post_proj = nn.Linear(hidden_dim, hidden_dim)
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.user_mental_proj, self.listener_mental_proj,
                     self.topic_proj, self.knowledge_proj,
                     self.situation_proj, self.post_proj]:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
    
    def forward(self, encoder_hidden, attention_mask, situation_hidden=None, situation_mask=None, **kwargs):
        batch_size = encoder_hidden.size(0)
        mask = attention_mask.unsqueeze(-1).float()
        context = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        hp = self.post_proj(context)

        # Eq. 6: hs = Mean-pooling(H^s)
        if situation_hidden is not None and situation_mask is not None:
            sit_mask = situation_mask.unsqueeze(-1).float()
            sit_pooled = (situation_hidden * sit_mask).sum(dim=1) / sit_mask.sum(dim=1).clamp(min=1e-9)
            hs = self.situation_proj(sit_pooled)
        else:
            hs = self.situation_proj(context)

        ht = self.topic_proj(context)
        he = self.knowledge_proj(context)
        user_mental = self.user_mental_proj(context).view(batch_size, 6, self.hidden_dim)
        listener_mental = self.listener_mental_proj(context).view(batch_size, 3, self.hidden_dim)

        return {'hp': hp, 'hs': hs, 'ht': ht, 'he': he,
                'user_mental': user_mental, 'listener_mental': listener_mental}


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    """
    PMPC Model - 논문 완전 구현
    
    구성요소:
    1. Cues Catcher: COMET + GloVe + DPR (or Lightweight fallback)
    2. Prompt Builder: Pe, Pd, Strategy Codebook
    3. Response Generator: BlenderBot + Prepend prompts
    """
    
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)

        hidden_dim = config.d_model
        self.hidden_dim = hidden_dim
        self.prompt_len = 20  # 논문 Implementation Details
        self.n_strategies = 8
        self.use_full_cues_catcher = USE_FULL_CUES_CATCHER

        # Placeholder for Cues Catcher — 외부 모델(COMET/DPR/GloVe)은
        # from_pretrained()가 _init_weights()로 가중치를 덮어쓰므로
        # __init__에서 로드하면 안 됨. load_external_models()에서 로드.
        self.cues_catcher = LightweightCuesCatcher(hidden_dim)

        # Prompt Builder
        self.prompt_builder = PMPCPromptBuilder(
            hidden_dim=hidden_dim,
            prompt_len=self.prompt_len,
            n_strategies=self.n_strategies,
        )

        self.strat_loss_weight = 1.0

        logger.info(f"PMPC Model initialized: hidden_dim={hidden_dim}, prompt_len={self.prompt_len}")

    def load_external_models(self):
        """
        from_pretrained() 이후에 호출하여 외부 모델을 로드.
        HuggingFace의 _fast_init이 __init__ 내에서 로드된 외부 모델
        가중치를 덮어쓰는 문제를 우회.
        """
        if not USE_FULL_CUES_CATCHER:
            logger.info("Using Lightweight Cues Catcher (no external models)")
            return

        try:
            self.cues_catcher = FullCuesCatcher(
                hidden_dim=self.hidden_dim,
                device='cpu',  # deploy_model()에서 GPU로 이동
                comet_path='./external_models/comet-atomic-2020',
                glove_path='./external_models/glove.6B.300d.txt',
                dpr_ctx_path='./external_models/dpr-ctx-encoder',
                dpr_q_path='./external_models/dpr-question-encoder',
                training_responses_path='./_reformat/train.txt',
                top_k_topics=10,
                top_l_knowledge=5,
            )
            self.use_full_cues_catcher = True
            logger.info("Full Cues Catcher (COMET + GloVe + DPR) loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Full Cues Catcher: {e}")
            logger.warning("Keeping Lightweight Cues Catcher")
            self.use_full_cues_catcher = False
    
    def _extract_text_from_batch(self, input_ids, encoded_info):
        """Extract text information from batch for Full Cues Catcher"""
        batch_size = input_ids.size(0)
        
        # Try to get text from encoded_info
        dialogue_histories = encoded_info.get('dialogue_histories', None)
        situations = encoded_info.get('situations', None)
        last_utterances = encoded_info.get('last_utterances', None)
        
        # If not available, decode from input_ids (fallback)
        if dialogue_histories is None and self.toker is not None:
            try:
                dialogue_histories = []
                for i in range(batch_size):
                    text = self.toker.decode(input_ids[i], skip_special_tokens=True)
                    dialogue_histories.append(text)
                    
                # Extract last utterance (simplified)
                if last_utterances is None:
                    last_utterances = []
                    for text in dialogue_histories:
                        # Get last part after [EOS] or just the whole text
                        parts = text.split('</s>')
                        last_utterances.append(parts[-1].strip() if parts else text)
                
                if situations is None:
                    situations = [''] * batch_size
            except Exception as e:
                logger.debug(f"Failed to decode text: {e}")
                dialogue_histories = None
                situations = None
                last_utterances = None
        
        return dialogue_histories, situations, last_utterances
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        strat_id=None,
        use_cache=None,
        return_dict=None,
        validation=False,
        **kwargs
    ):
        assert self.toker is not None
        
        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None)
        if validation:
            labels[:, 0] = -100
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # === Inference: 표준 forward ===
        if not self.training and not validation:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
            
            return Seq2SeqLMOutput(
                loss=None, logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        # === Training/Validation ===
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Step 1: 원본 입력 임베딩
        input_embeds = self.model.encoder.embed_tokens(input_ids) * self.model.encoder.embed_scale
        
        # Step 2: 첫 번째 encoder pass (cues 추출용)
        encoder_outputs_init = self.model.encoder(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden = encoder_outputs_init.last_hidden_state
        
        # Step 3: Encode situation separately for Eq. 6: hs = Mean-pooling(H^s)
        situation_hidden = None
        sit_mask = None
        situation_ids = encoded_info.get('situation_ids', None)
        sit_mask_raw = encoded_info.get('situation_mask', None)
        if situation_ids is not None and sit_mask_raw is not None:
            situation_ids = situation_ids.to(device)
            sit_mask = sit_mask_raw.to(device)
            sit_embeds = self.model.encoder.embed_tokens(situation_ids) * self.model.encoder.embed_scale
            sit_enc_out = self.model.encoder(
                input_ids=None,
                inputs_embeds=sit_embeds,
                attention_mask=sit_mask,
                return_dict=True,
            )
            situation_hidden = sit_enc_out.last_hidden_state

        # Step 4: Cues 추출
        if self.use_full_cues_catcher:
            # Extract text for Full Cues Catcher
            dialogue_histories, situations, last_utterances = self._extract_text_from_batch(
                input_ids, encoded_info
            )
            cues = self.cues_catcher(
                encoder_hidden=encoder_hidden,
                attention_mask=attention_mask,
                dialogue_histories=dialogue_histories,
                situations=situations,
                last_utterances=last_utterances,
                situation_hidden=situation_hidden,
                situation_mask=sit_mask,
            )
        else:
            cues = self.cues_catcher(encoder_hidden, attention_mask,
                                     situation_hidden=situation_hidden,
                                     situation_mask=sit_mask)
        
        # NaN 체크: 어디서 NaN이 발생하는지 추적
        _nan_count = getattr(self, '_nan_count', 0)
        _nan_logged = getattr(self, '_nan_logged', False)
        for cue_name in ['hp', 'hs', 'ht', 'he', 'user_mental', 'listener_mental']:
            cue = cues[cue_name]
            if torch.isnan(cue).any() or torch.isinf(cue).any():
                _nan_count += 1
                if not _nan_logged:
                    # 첫 NaN 발생시 상세 정보 출력
                    logger.warning(
                        f"NaN/Inf in {cue_name}! "
                        f"shape={cue.shape}, nan_count={torch.isnan(cue).sum()}, "
                        f"inf_count={torch.isinf(cue).sum()}"
                    )
                    # Encoder hidden state도 확인
                    logger.warning(
                        f"encoder_hidden: nan={torch.isnan(encoder_hidden).any()}, "
                        f"mean={encoder_hidden.mean():.6f}, std={encoder_hidden.std():.6f}"
                    )
                    self._nan_logged = True
                cues[cue_name] = torch.zeros_like(cue)
        self._nan_count = _nan_count
        if _nan_count > 0 and _nan_count % 100 == 0:
            logger.warning(f"Total NaN occurrences so far: {_nan_count}")

        # Step 4: Pe 계산 (Eq. 9)
        hx = self.prompt_builder.compute_hx(cues['user_mental'])
        Pe = self.prompt_builder.compute_Pe(cues['hp'], cues['hs'], hx)  # (batch, 20, hidden)
        
        # Pe NaN 체크
        if torch.isnan(Pe).any() or torch.isinf(Pe).any():
            logger.warning("NaN/Inf in Pe, using zero prompt")
            Pe = torch.zeros_like(Pe)
        
        # Step 5: Pe를 encoder input 앞에 PREPEND + [CLS] 삽입 (논문 Eq. 13)
        # Eq. 13: Enc(e1,...,en, [CLS], μ1, [EOS], ...)
        # Scale Pe to match input embedding scale (embed_scale = sqrt(d_model) ≈ 22.6)
        Pe = Pe * self.model.encoder.embed_scale
        # [CLS] token embedding between Pe and input_embeds
        cls_token_id = torch.tensor([[self.config.bos_token_id]], device=device).expand(batch_size, -1)
        cls_embeds = self.model.encoder.embed_tokens(cls_token_id) * self.model.encoder.embed_scale  # (batch, 1, hidden)
        enhanced_embeds = torch.cat([Pe, cls_embeds, input_embeds], dim=1)  # (batch, 20+1+seq, hidden)

        # Attention mask 확장
        prompt_and_cls_mask = torch.ones(batch_size, self.prompt_len + 1, device=device, dtype=attention_mask.dtype)
        enhanced_mask = torch.cat([prompt_and_cls_mask, attention_mask], dim=1)  # (batch, 20+1+seq)

        # Position IDs 명시적 할당 (0부터 시작)
        seq_len_with_prompt = enhanced_embeds.size(1)
        position_ids = torch.arange(seq_len_with_prompt, device=device).unsqueeze(0).expand(batch_size, -1)

        # Truncate if exceeds max_position_embeddings (512)
        max_pos = self.config.max_position_embeddings
        if seq_len_with_prompt > max_pos:
            enhanced_embeds = enhanced_embeds[:, :max_pos, :]
            enhanced_mask = enhanced_mask[:, :max_pos]
            position_ids = position_ids[:, :max_pos]

        # Step 6: Enhanced encoder pass
        encoder_outputs_enhanced = self.model.encoder(
            input_ids=None,
            inputs_embeds=enhanced_embeds,
            attention_mask=enhanced_mask,
            return_dict=True,
        )
        encoder_hidden_enhanced = encoder_outputs_enhanced.last_hidden_state

        # Step 7: Strategy prediction (Eq. 14) — H_cls is at the [CLS] position
        Hcls = encoder_hidden_enhanced[:, self.prompt_len, :]
        strategy_logits, strategy_probs = self.prompt_builder.predict_strategy(Hcls)
        
        # Step 8: hstrat 및 Pd 계산 (Eq. 11, 15)
        hstrat = self.prompt_builder.compute_hstrat(strategy_probs)
        ho = self.prompt_builder.compute_ho(cues['listener_mental'])
        Pd = self.prompt_builder.compute_Pd(cues['hp'], cues['ht'], cues['he'], ho, hstrat)
        
        # Pd NaN 체크
        if torch.isnan(Pd).any() or torch.isinf(Pd).any():
            logger.warning("NaN/Inf in Pd, using zero prompt")
            Pd = torch.zeros_like(Pd)
        
        # Step 9: Decoder - Pd를 앞에 PREPEND (논문 Eq. 16)
        # Scale Pd to match decoder embedding scale
        Pd = Pd * self.model.decoder.embed_scale
        decoder_embeds = self.model.decoder.embed_tokens(decoder_input_ids) * self.model.decoder.embed_scale
        enhanced_decoder_embeds = torch.cat([Pd, decoder_embeds], dim=1)  # (batch, 20+dec_seq, hidden)
        
        # Decoder seq length 확인
        dec_seq_len_with_prompt = enhanced_decoder_embeds.size(1)
        if dec_seq_len_with_prompt > max_pos:
            enhanced_decoder_embeds = enhanced_decoder_embeds[:, :max_pos, :]
        
        # Step 10: Decoder forward
        decoder_outputs = self.model.decoder(
            input_ids=None,
            inputs_embeds=enhanced_decoder_embeds,
            encoder_hidden_states=encoder_hidden_enhanced,
            encoder_attention_mask=enhanced_mask,
            use_cache=False,
            return_dict=True,
        )
        
        # Step 11: LM head
        lm_logits_full = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        
        # Step 12: Pd 위치 제외하고 loss 계산
        lm_logits = lm_logits_full[:, self.prompt_len:, :]  # (batch, dec_seq, vocab)
        
        if validation:
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

        # Step 13: Loss 계산
        lm_logits = lm_logits.contiguous()
        loss = F.cross_entropy(lm_logits.reshape(-1, lm_logits.size(-1)), labels.reshape(-1), reduction='none')
        loss = loss.reshape(labels.size(0), labels.size(1))
        label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
        masked_lm_loss = torch.sum(loss) / torch.sum(label_size).clamp(min=1.0)
        ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float().clamp(min=1.0)))

        # Strategy loss
        strat_loss = torch.tensor(0.0, device=device)
        if strat_id is not None:
            valid_mask = (strat_id >= 0) & (strat_id < self.n_strategies)
            if valid_mask.any():
                strat_loss = F.cross_entropy(
                    strategy_logits[valid_mask],
                    strat_id[valid_mask].clamp(0, self.n_strategies - 1)
                )

        if self.training:
            total_loss = masked_lm_loss + self.strat_loss_weight * strat_loss
            return {'all': total_loss, 'ppl': ppl_value, 'lm': masked_lm_loss, 'strat': strat_loss}

        return loss, label_size

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=None,
        **kwargs
    ):
        """추론 시에도 훈련과 동일하게 Pe 적용 + strategy_predictor 사용"""
        assert not self.training
        assert self.toker is not None
        
        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Step 1: 원본 입력 임베딩
        input_embeds = self.model.encoder.embed_tokens(input_ids) * self.model.encoder.embed_scale
        
        # Step 2: 첫 번째 encoder pass (cues 추출용)
        encoder_outputs_init = self.model.encoder(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden = encoder_outputs_init.last_hidden_state
        
        # Step 3: Encode situation separately for Eq. 6
        situation_hidden = None
        sit_mask = None
        situation_ids = encoded_info.get('situation_ids', None)
        sit_mask_raw = encoded_info.get('situation_mask', None)
        if situation_ids is not None and sit_mask_raw is not None:
            situation_ids = situation_ids.to(device)
            sit_mask = sit_mask_raw.to(device)
            sit_embeds = self.model.encoder.embed_tokens(situation_ids) * self.model.encoder.embed_scale
            sit_enc_out = self.model.encoder(
                input_ids=None,
                inputs_embeds=sit_embeds,
                attention_mask=sit_mask,
                return_dict=True,
            )
            situation_hidden = sit_enc_out.last_hidden_state

        # Step 4: Cues 추출
        if self.use_full_cues_catcher:
            dialogue_histories, situations, last_utterances = self._extract_text_from_batch(
                input_ids, encoded_info
            )
            cues = self.cues_catcher(
                encoder_hidden=encoder_hidden,
                attention_mask=attention_mask,
                dialogue_histories=dialogue_histories,
                situations=situations,
                last_utterances=last_utterances,
                situation_hidden=situation_hidden,
                situation_mask=sit_mask,
            )
        else:
            cues = self.cues_catcher(encoder_hidden, attention_mask,
                                     situation_hidden=situation_hidden,
                                     situation_mask=sit_mask)

        # NaN 체크 및 안전 처리 (generate에서도 필요)
        def safe_cue_gen(cue, fallback_shape, name="cue"):
            if torch.isnan(cue).any() or torch.isinf(cue).any():
                return torch.zeros(fallback_shape, device=device, dtype=cue.dtype)
            return cue

        cues['hp'] = safe_cue_gen(cues['hp'], cues['hp'].shape, 'hp')
        cues['hs'] = safe_cue_gen(cues['hs'], cues['hs'].shape, 'hs')
        cues['ht'] = safe_cue_gen(cues['ht'], cues['ht'].shape, 'ht')
        cues['he'] = safe_cue_gen(cues['he'], cues['he'].shape, 'he')
        cues['user_mental'] = safe_cue_gen(cues['user_mental'], cues['user_mental'].shape, 'user_mental')
        cues['listener_mental'] = safe_cue_gen(cues['listener_mental'], cues['listener_mental'].shape, 'listener_mental')
        
        # Step 5: Pe 계산
        hx = self.prompt_builder.compute_hx(cues['user_mental'])
        Pe = self.prompt_builder.compute_Pe(cues['hp'], cues['hs'], hx)

        # Pe NaN 체크
        if torch.isnan(Pe).any() or torch.isinf(Pe).any():
            Pe = torch.zeros_like(Pe)

        # Step 6: Pe + [CLS] 를 encoder input 앞에 PREPEND (논문 Eq. 13)
        Pe = Pe * self.model.encoder.embed_scale  # Scale to match input embeddings
        cls_token_id = torch.tensor([[self.config.bos_token_id]], device=device).expand(batch_size, -1)
        cls_embeds = self.model.encoder.embed_tokens(cls_token_id) * self.model.encoder.embed_scale
        enhanced_embeds = torch.cat([Pe, cls_embeds, input_embeds], dim=1)
        prompt_and_cls_mask = torch.ones(batch_size, self.prompt_len + 1, device=device, dtype=attention_mask.dtype)
        enhanced_mask = torch.cat([prompt_and_cls_mask, attention_mask], dim=1)

        # Truncate if needed
        max_pos = self.config.max_position_embeddings
        if enhanced_embeds.size(1) > max_pos:
            enhanced_embeds = enhanced_embeds[:, :max_pos, :]
            enhanced_mask = enhanced_mask[:, :max_pos]

        # Step 7: Enhanced encoder pass
        encoder_outputs_enhanced = self.model.encoder(
            input_ids=None,
            inputs_embeds=enhanced_embeds,
            attention_mask=enhanced_mask,
            return_dict=True,
        )
        encoder_hidden_enhanced = encoder_outputs_enhanced.last_hidden_state

        # encoder_hidden_enhanced NaN 체크
        if torch.isnan(encoder_hidden_enhanced).any() or torch.isinf(encoder_hidden_enhanced).any():
            logger.warning("NaN in encoder_hidden_enhanced during generate, using fallback")
            encoder_hidden_enhanced = torch.zeros_like(encoder_hidden_enhanced)

        # Step 8: Strategy prediction (Eq. 14) — H_cls at [CLS] position
        Hcls = encoder_hidden_enhanced[:, self.prompt_len, :]

        # NaN 체크
        if torch.isnan(Hcls).any() or torch.isinf(Hcls).any():
            Hcls = torch.zeros_like(Hcls)

        strategy_logits, strategy_probs = self.prompt_builder.predict_strategy(Hcls)

        # strategy_logits/probs NaN 체크
        if torch.isnan(strategy_logits).any() or torch.isinf(strategy_logits).any():
            strategy_logits = torch.zeros_like(strategy_logits)
            strategy_probs = torch.ones(batch_size, self.n_strategies, device=device) / self.n_strategies

        # 전략 선택 (argmax)
        pred_strat = torch.argmax(strategy_logits, dim=-1)
        pred_top1 = torch.topk(strategy_logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(strategy_logits, k=3, dim=-1)[1]

        encoded_info.update({
            'pred_strat_id': pred_strat,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': strategy_probs
        })

        # Step 9: Pd 계산 (논문 Eq. 11, 15, 16 — 추론에서도 Pd 적용)
        hstrat = self.prompt_builder.compute_hstrat(strategy_probs)
        ho = self.prompt_builder.compute_ho(cues['listener_mental'])
        Pd = self.prompt_builder.compute_Pd(cues['hp'], cues['ht'], cues['he'], ho, hstrat)

        if torch.isnan(Pd).any() or torch.isinf(Pd).any():
            Pd = torch.zeros_like(Pd)
        # Scale Pd to match decoder embedding scale
        Pd = Pd * self.model.decoder.embed_scale

        # decoder_input_ids에 전략 토큰 추가
        decoder_input_ids = torch.cat([
            decoder_input_ids,
            encoded_info['pred_strat_id'][..., None] + len(self.toker) - 8
        ], dim=-1)

        # Step 10: Custom autoregressive generation with Pd prepended (논문 Eq. 16)
        assert 'max_length' in kwargs
        max_length = kwargs['max_length']
        temperature = kwargs.get('temperature', TEMPERATURE)
        top_k = kwargs.get('top_k', 0)
        top_p = kwargs.get('top_p', 1.0)
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        eos_token_id = self.config.eos_token_id

        # Strategy token IDs to suppress (bad_words)
        bad_word_ids = set()
        if len(self.toker) > self.toker.vocab_size:
            bad_word_ids = set(range(self.toker.vocab_size, len(self.toker)))

        generated_tokens = decoder_input_ids  # (batch, 2) — [bos, strat]

        for step in range(max_length):
            # Embed current decoder tokens
            dec_embeds = self.model.decoder.embed_tokens(generated_tokens) * self.model.decoder.embed_scale
            # Prepend Pd (논문 Eq. 16)
            dec_embeds_with_pd = torch.cat([Pd, dec_embeds], dim=1)

            # Truncate decoder if needed
            if dec_embeds_with_pd.size(1) > max_pos:
                dec_embeds_with_pd = dec_embeds_with_pd[:, :max_pos, :]

            # Decoder forward
            decoder_outputs = self.model.decoder(
                input_ids=None,
                inputs_embeds=dec_embeds_with_pd,
                encoder_hidden_states=encoder_hidden_enhanced,
                encoder_attention_mask=enhanced_mask,
                use_cache=False,
                return_dict=True,
            )

            # Get logits for the last position (skip Pd positions)
            next_token_logits = self.lm_head(decoder_outputs.last_hidden_state[:, -1, :]) + self.final_logits_bias
            next_token_logits = next_token_logits.squeeze(1) if next_token_logits.dim() == 3 else next_token_logits

            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for prev_token in generated_tokens[i].tolist():
                        if next_token_logits[i, prev_token] > 0:
                            next_token_logits[i, prev_token] /= repetition_penalty
                        else:
                            next_token_logits[i, prev_token] *= repetition_penalty

            # Suppress bad words (strategy tokens)
            for bad_id in bad_word_ids:
                next_token_logits[:, bad_id] = -float('inf')

            # Temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-k / top-p filtering
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            # Check EOS
            if (next_token == eos_token_id).all():
                break

        return encoded_info, generated_tokens[:, decoder_input_ids.size(1):]
