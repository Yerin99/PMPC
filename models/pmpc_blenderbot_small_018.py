# coding=utf-8
"""
PMPC + 018 Method Integration

Base: pmpc_blenderbot_small.py (PMPC, IEEE TCSS 2025)
Added: LoRA Expert Routing, Scheduled Sampling (via strategy_predictor, no probe)

Key difference from DKFPE 018:
- No separate probe head; uses PMPC's existing strategy_predictor
- SS logits source: strategy_logits.detach() (not probe_logits)
- Pd (hstrat) pipeline unchanged; LoRA routing is the only 018 addition
- Ensemble uses compute_strategy_distribution() instead of probe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
from models.lora_expert_layers import LoRAExpertContext, wrap_modules_with_lora
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
    """PMPC Prompt Builder - Section III-B (identical to base PMPC)"""

    def __init__(self, hidden_dim: int = 512, prompt_len: int = 20, n_strategies: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prompt_len = prompt_len
        self.n_strategies = n_strategies

        # Eq. 10: hx = Wx[Hc1 ... Hc6]
        self.Wx = nn.Linear(hidden_dim * 6, hidden_dim)
        # Eq. 12: ho = Wo[Hc7 Hc8 Hc9]
        self.Wo = nn.Linear(hidden_dim * 3, hidden_dim)

        # Eq. 9: Pe = MLPe(hp hs hx)
        self.MLPe = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, prompt_len * hidden_dim),
        )

        # Eq. 11: Pd = MLPd(hp ht he ho hstrat)
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
    PMPC + 018 Method

    Components:
    1. Cues Catcher: COMET + GloVe + DPR (or Lightweight fallback) — PMPC original
    2. Prompt Builder: Pe, Pd, Strategy Codebook — PMPC original
    3. Response Generator: BlenderBot + Prepend prompts — PMPC original
    4. LoRA Expert Routing on decoder (8 experts) + shared LoRA on encoder — 018
    5. Scheduled Sampling via strategy_predictor — 018 (no probe)
    """

    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)

        hidden_dim = config.d_model
        self.hidden_dim = hidden_dim
        self.prompt_len = 20
        self.n_strategies = 8
        self.use_full_cues_catcher = USE_FULL_CUES_CATCHER

        self.cues_catcher = LightweightCuesCatcher(hidden_dim)

        self.prompt_builder = PMPCPromptBuilder(
            hidden_dim=hidden_dim,
            prompt_len=self.prompt_len,
            n_strategies=self.n_strategies,
        )

        self.strat_loss_weight = 1.0

        # 018 fields
        self._lora_initialized = False
        self._decoder_lora_context = LoRAExpertContext()
        self._lora_num_experts = 8
        self._scheduled_sampler = None
        self._no_strat_in_seq = False

        logger.info(f"PMPC 018 Model initialized: hidden_dim={hidden_dim}, prompt_len={self.prompt_len}")

    def load_external_models(self):
        """Load COMET/DPR/GloVe after from_pretrained()."""
        if not USE_FULL_CUES_CATCHER:
            logger.info("Using Lightweight Cues Catcher (no external models)")
            return

        try:
            self.cues_catcher = FullCuesCatcher(
                hidden_dim=self.hidden_dim,
                device='cpu',
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

    def apply_runtime_config(self, runtime_config):
        """Initialize LoRA expert routing from config JSON. No probe head."""
        if self._lora_initialized:
            return
        if self.toker is None:
            raise RuntimeError("Tokenizer must be tied before LoRA initialization")

        self._lora_num_experts = int(runtime_config.get("lora_num_experts", 8))
        lora_rank = int(runtime_config.get("lora_rank", 8))
        lora_alpha = int(runtime_config.get("lora_alpha", 16))
        lora_dropout = float(runtime_config.get("lora_dropout", 0.05))
        lora_base_trainable = bool(runtime_config.get("lora_base_trainable", True))
        lora_use_encoder_shared = bool(runtime_config.get("lora_use_encoder_shared", True))
        lora_total_experts = self._lora_num_experts + 1  # +1 for shared slot

        target_proj = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
        is_target = lambda n, m: isinstance(m, nn.Linear) and n.split(".")[-1] in target_proj

        # Decoder: expert LoRA
        decoder_wrapped = wrap_modules_with_lora(
            root=self.model.decoder,
            predicate=is_target,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            num_experts=lora_total_experts,
            context=self._decoder_lora_context,
        )
        if not decoder_wrapped:
            raise RuntimeError("No decoder modules were wrapped by LoRA")

        # Encoder: shared LoRA (no expert routing)
        if lora_use_encoder_shared:
            encoder_wrapped = wrap_modules_with_lora(
                root=self.model.encoder,
                predicate=is_target,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                num_experts=None,
                context=None,
            )
            if not encoder_wrapped:
                raise RuntimeError("No encoder modules were wrapped by shared LoRA")

        # Trainability
        if lora_base_trainable:
            for p in self.parameters():
                p.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = False
            for name, p in self.named_parameters():
                if ".lora_a" in name or ".lora_b" in name:
                    p.requires_grad = True

        # B condition
        self._no_strat_in_seq = bool(runtime_config.get("no_strat_in_seq", False))

        self._lora_initialized = True
        logger.info(f"LoRA initialized: {self._lora_num_experts} experts, "
                     f"rank={lora_rank}, no_strat_in_seq={self._no_strat_in_seq}")

    def set_scheduled_sampler(self, sampler):
        self._scheduled_sampler = sampler

    # --- Expert routing helpers ---
    def _set_expert_ids(self, strat_id: torch.Tensor):
        self._decoder_lora_context.expert_ids = strat_id.long()
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = False

    def _disable_expert_lora(self):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = True

    def _clear_expert_context(self):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = False

    def _validate_strategy_ids(self, strat_id: torch.Tensor, batch_size: int):
        model_device = next(self.parameters()).device
        if not isinstance(strat_id, torch.Tensor):
            strat_id = torch.tensor(strat_id, device=model_device)
        strat_id = strat_id.long()
        if strat_id.dim() == 0:
            strat_id = strat_id.unsqueeze(0)
        if strat_id.dim() != 1 or strat_id.size(0) != batch_size:
            raise RuntimeError("strategy id must be [batch] and match batch size")
        if (strat_id < 0).any() or (strat_id >= self._lora_num_experts).any():
            raise RuntimeError(f"strategy id must be in [0, {self._lora_num_experts - 1}]")
        return strat_id

    def _extract_text_from_batch(self, input_ids, encoded_info):
        """Extract text information from batch for Full Cues Catcher"""
        batch_size = input_ids.size(0)

        dialogue_histories = encoded_info.get('dialogue_histories', None)
        situations = encoded_info.get('situations', None)
        last_utterances = encoded_info.get('last_utterances', None)

        if dialogue_histories is None and self.toker is not None:
            try:
                dialogue_histories = []
                for i in range(batch_size):
                    text = self.toker.decode(input_ids[i], skip_special_tokens=True)
                    dialogue_histories.append(text)

                if last_utterances is None:
                    last_utterances = []
                    for text in dialogue_histories:
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

    # --- PMPC pipeline helpers (shared by forward and generate) ---
    def _run_cues_extraction(self, input_ids, attention_mask, encoded_info, device):
        """Steps 1-4: encoder pass, situation encoding, cues extraction."""
        # Step 1: Input embeddings
        input_embeds = self.model.encoder.embed_tokens(input_ids) * self.model.encoder.embed_scale

        # Step 2: First encoder pass
        encoder_outputs_init = self.model.encoder(
            input_ids=None, inputs_embeds=input_embeds,
            attention_mask=attention_mask, return_dict=True,
        )
        encoder_hidden = encoder_outputs_init.last_hidden_state

        # Step 3: Encode situation separately (Eq. 6)
        situation_hidden = None
        sit_mask = None
        situation_ids = encoded_info.get('situation_ids', None)
        sit_mask_raw = encoded_info.get('situation_mask', None)
        if situation_ids is not None and sit_mask_raw is not None:
            situation_ids = situation_ids.to(device)
            sit_mask = sit_mask_raw.to(device)
            sit_embeds = self.model.encoder.embed_tokens(situation_ids) * self.model.encoder.embed_scale
            sit_enc_out = self.model.encoder(
                input_ids=None, inputs_embeds=sit_embeds,
                attention_mask=sit_mask, return_dict=True,
            )
            situation_hidden = sit_enc_out.last_hidden_state

        # Step 4: Cues extraction
        if self.use_full_cues_catcher:
            dialogue_histories, situations, last_utterances = self._extract_text_from_batch(
                input_ids, encoded_info
            )
            cues = self.cues_catcher(
                encoder_hidden=encoder_hidden, attention_mask=attention_mask,
                dialogue_histories=dialogue_histories, situations=situations,
                last_utterances=last_utterances,
                situation_hidden=situation_hidden, situation_mask=sit_mask,
            )
        else:
            cues = self.cues_catcher(encoder_hidden, attention_mask,
                                     situation_hidden=situation_hidden,
                                     situation_mask=sit_mask)

        return input_embeds, cues

    def _safe_cues(self, cues):
        """NaN/Inf check for cues."""
        _nan_count = getattr(self, '_nan_count', 0)
        _nan_logged = getattr(self, '_nan_logged', False)
        for cue_name in ['hp', 'hs', 'ht', 'he', 'user_mental', 'listener_mental']:
            cue = cues[cue_name]
            if torch.isnan(cue).any() or torch.isinf(cue).any():
                _nan_count += 1
                if not _nan_logged:
                    logger.warning(f"NaN/Inf in {cue_name}! shape={cue.shape}")
                    self._nan_logged = True
                cues[cue_name] = torch.zeros_like(cue)
        self._nan_count = _nan_count
        return cues

    def _run_enhanced_encoder(self, input_embeds, cues, attention_mask, batch_size, device):
        """Steps 5-7: Pe computation, enhanced encoder, strategy prediction."""
        # Step 5: Pe computation (Eq. 9)
        hx = self.prompt_builder.compute_hx(cues['user_mental'])
        Pe = self.prompt_builder.compute_Pe(cues['hp'], cues['hs'], hx)

        if torch.isnan(Pe).any() or torch.isinf(Pe).any():
            Pe = torch.zeros_like(Pe)

        # Step 6: Pe + [CLS] prepend (Eq. 13)
        Pe = Pe * self.model.encoder.embed_scale
        cls_token_id = torch.tensor([[self.config.bos_token_id]], device=device).expand(batch_size, -1)
        cls_embeds = self.model.encoder.embed_tokens(cls_token_id) * self.model.encoder.embed_scale
        enhanced_embeds = torch.cat([Pe, cls_embeds, input_embeds], dim=1)

        prompt_and_cls_mask = torch.ones(batch_size, self.prompt_len + 1, device=device, dtype=attention_mask.dtype)
        enhanced_mask = torch.cat([prompt_and_cls_mask, attention_mask], dim=1)

        # Truncate if exceeds max_position_embeddings
        max_pos = self.config.max_position_embeddings
        if enhanced_embeds.size(1) > max_pos:
            enhanced_embeds = enhanced_embeds[:, :max_pos, :]
            enhanced_mask = enhanced_mask[:, :max_pos]

        # Enhanced encoder pass
        encoder_outputs_enhanced = self.model.encoder(
            input_ids=None, inputs_embeds=enhanced_embeds,
            attention_mask=enhanced_mask, return_dict=True,
        )
        encoder_hidden_enhanced = encoder_outputs_enhanced.last_hidden_state

        # Step 7: Strategy prediction (Eq. 14)
        Hcls = encoder_hidden_enhanced[:, self.prompt_len, :]
        strategy_logits, strategy_probs = self.prompt_builder.predict_strategy(Hcls)

        return encoder_hidden_enhanced, enhanced_mask, strategy_logits, strategy_probs

    @torch.no_grad()
    def compute_strategy_distribution(self, input_ids, attention_mask, **kwargs):
        """Compute strategy_probs for ensemble inference.

        Runs steps 1-7 of the PMPC pipeline and returns strategy_probs [batch, 8].
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Steps 1-4: Cues extraction
        input_embeds, cues = self._run_cues_extraction(input_ids, attention_mask, kwargs, device)
        cues = self._safe_cues(cues)

        # Steps 5-7: Enhanced encoder + strategy prediction
        _, _, strategy_logits, strategy_probs = self._run_enhanced_encoder(
            input_embeds, cues, attention_mask, batch_size, device
        )

        # NaN safety
        if torch.isnan(strategy_probs).any() or torch.isinf(strategy_probs).any():
            strategy_probs = torch.ones(batch_size, self.n_strategies, device=device) / self.n_strategies

        return strategy_probs

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
            # A condition: labels[0] is strategy token -> mask it
            # B condition: labels[0] is real response token -> don't mask
            if not self._no_strat_in_seq:
                labels[:, 0] = -100

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # === Inference: standard forward (HuggingFace generate internal calls) ===
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

        if strat_id is not None:
            strat_id = self._validate_strategy_ids(strat_id, batch_size)

        # Steps 1-4: Cues extraction
        input_embeds, cues = self._run_cues_extraction(input_ids, attention_mask, encoded_info, device)
        cues = self._safe_cues(cues)

        # Steps 5-7: Enhanced encoder + strategy prediction
        encoder_hidden_enhanced, enhanced_mask, strategy_logits, strategy_probs = self._run_enhanced_encoder(
            input_embeds, cues, attention_mask, batch_size, device
        )

        # Step 7.5 (018): Scheduled sampling -> routing_strat_id
        routing_strat_id = strat_id  # default: GT strategy
        if self.training and self._scheduled_sampler is not None and strat_id is not None:
            routing_strat_id = self._scheduled_sampler.sample_strategy_tokens(
                gt_strategy_ids=strat_id,
                strategy_logits=strategy_logits.detach(),
            )

        # Step 8: hstrat + Pd computation (UNCHANGED from base PMPC)
        hstrat = self.prompt_builder.compute_hstrat(strategy_probs)
        ho = self.prompt_builder.compute_ho(cues['listener_mental'])
        Pd = self.prompt_builder.compute_Pd(cues['hp'], cues['ht'], cues['he'], ho, hstrat)

        if torch.isnan(Pd).any() or torch.isinf(Pd).any():
            Pd = torch.zeros_like(Pd)

        # Step 8.5 (018): Set expert routing
        if routing_strat_id is not None:
            self._set_expert_ids(routing_strat_id)
        else:
            self._disable_expert_lora()

        try:
            # Step 9: Decoder with Pd prepend (Eq. 16) — LoRA active
            Pd_scaled = Pd * self.model.decoder.embed_scale
            decoder_embeds = self.model.decoder.embed_tokens(decoder_input_ids) * self.model.decoder.embed_scale
            enhanced_decoder_embeds = torch.cat([Pd_scaled, decoder_embeds], dim=1)

            max_pos = self.config.max_position_embeddings
            if enhanced_decoder_embeds.size(1) > max_pos:
                enhanced_decoder_embeds = enhanced_decoder_embeds[:, :max_pos, :]

            decoder_outputs = self.model.decoder(
                input_ids=None,
                inputs_embeds=enhanced_decoder_embeds,
                encoder_hidden_states=encoder_hidden_enhanced,
                encoder_attention_mask=enhanced_mask,
                use_cache=False,
                return_dict=True,
            )

            # Step 10: LM head
            lm_logits_full = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
            lm_logits = lm_logits_full[:, self.prompt_len:, :]  # skip Pd positions

            if validation:
                lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

            # Loss computation
            lm_logits = lm_logits.contiguous()
            loss = F.cross_entropy(lm_logits.reshape(-1, lm_logits.size(-1)), labels.reshape(-1), reduction='none')
            loss = loss.reshape(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size).clamp(min=1.0)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float().clamp(min=1.0)))

            # Strategy loss (strategy_predictor's own loss)
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
        finally:
            # Step 9.5 (018): Clear expert context
            self._clear_expert_context()

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=None,
        **kwargs
    ):
        assert not self.training
        assert self.toker is not None

        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)
        device = input_ids.device

        # Steps 1-4: Cues extraction
        input_embeds, cues = self._run_cues_extraction(input_ids, attention_mask, encoded_info, device)

        # NaN safety for generate
        for cue_name in ['hp', 'hs', 'ht', 'he', 'user_mental', 'listener_mental']:
            cue = cues[cue_name]
            if torch.isnan(cue).any() or torch.isinf(cue).any():
                cues[cue_name] = torch.zeros_like(cue)

        # Steps 5-7: Enhanced encoder + strategy prediction
        encoder_hidden_enhanced, enhanced_mask, strategy_logits, strategy_probs = self._run_enhanced_encoder(
            input_embeds, cues, attention_mask, batch_size, device
        )

        # NaN safety
        if torch.isnan(encoder_hidden_enhanced).any() or torch.isinf(encoder_hidden_enhanced).any():
            encoder_hidden_enhanced = torch.zeros_like(encoder_hidden_enhanced)
        if torch.isnan(strategy_logits).any() or torch.isinf(strategy_logits).any():
            strategy_logits = torch.zeros_like(strategy_logits)
            strategy_probs = torch.ones(batch_size, self.n_strategies, device=device) / self.n_strategies

        # Step 7.5 (018): Strategy selection
        strat_id = encoded_info.get('strat_id', None)
        if strat_id is not None:
            # External strat_id provided (ensemble)
            if not isinstance(strat_id, torch.Tensor):
                strat_id = torch.tensor(strat_id, device=device)
            strat_id = strat_id.long()
            if strat_id.dim() == 0:
                strat_id = strat_id.unsqueeze(0)
            strat_id = self._validate_strategy_ids(strat_id, batch_size)
        else:
            # Use strategy_predictor's argmax (PMPC default)
            strat_id = torch.argmax(strategy_logits, dim=-1)

        pred_top1 = torch.topk(strategy_logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(strategy_logits, k=3, dim=-1)[1]

        encoded_info.update({
            'pred_strat_id': strat_id,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': strategy_probs,
        })

        # Step 8: Pd computation
        # When external strat_id: use hard lookup for hstrat
        # When self-predicted: use soft (matmul) — PMPC default
        external_strat = encoded_info.get('strat_id', None) is not None
        if external_strat:
            hstrat = self.prompt_builder.Estrat(strat_id)
        else:
            hstrat = self.prompt_builder.compute_hstrat(strategy_probs)

        ho = self.prompt_builder.compute_ho(cues['listener_mental'])
        Pd = self.prompt_builder.compute_Pd(cues['hp'], cues['ht'], cues['he'], ho, hstrat)

        if torch.isnan(Pd).any() or torch.isinf(Pd).any():
            Pd = torch.zeros_like(Pd)
        Pd = Pd * self.model.decoder.embed_scale

        # Step 9 (B condition): skip strategy token append
        if not self._no_strat_in_seq:
            decoder_input_ids = torch.cat([
                decoder_input_ids,
                strat_id[..., None] + len(self.toker) - 8
            ], dim=-1)

        # Step 8.5 (018): Set expert routing
        self._set_expert_ids(strat_id)

        try:
            # Step 10: Custom autoregressive generation with Pd prepended (Eq. 16)
            assert 'max_length' in kwargs
            max_length = kwargs['max_length']
            temperature = kwargs.get('temperature', TEMPERATURE)
            top_k = kwargs.get('top_k', 0)
            top_p = kwargs.get('top_p', 1.0)
            repetition_penalty = kwargs.get('repetition_penalty', 1.0)
            eos_token_id = self.config.eos_token_id

            bad_word_ids = set()
            if len(self.toker) > self.toker.vocab_size:
                bad_word_ids = set(range(self.toker.vocab_size, len(self.toker)))

            generated_tokens = decoder_input_ids
            max_pos = self.config.max_position_embeddings

            for step in range(max_length):
                dec_embeds = self.model.decoder.embed_tokens(generated_tokens) * self.model.decoder.embed_scale
                dec_embeds_with_pd = torch.cat([Pd, dec_embeds], dim=1)

                if dec_embeds_with_pd.size(1) > max_pos:
                    dec_embeds_with_pd = dec_embeds_with_pd[:, :max_pos, :]

                decoder_outputs = self.model.decoder(
                    input_ids=None,
                    inputs_embeds=dec_embeds_with_pd,
                    encoder_hidden_states=encoder_hidden_enhanced,
                    encoder_attention_mask=enhanced_mask,
                    use_cache=False,
                    return_dict=True,
                )

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

                # Suppress strategy tokens
                for bad_id in bad_word_ids:
                    next_token_logits[:, bad_id] = -float('inf')

                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if SAMPLE:
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

                if (next_token == eos_token_id).all():
                    break

            return encoded_info, generated_tokens[:, decoder_input_ids.size(1):]
        finally:
            # Step 10.5 (018): Clear expert context
            self._clear_expert_context()
