"""
Model Wrapper for Latent Reasoning.

Provides a unified interface over HuggingFace AutoModelForCausalLM,
supporting GPT-2, Qwen2.5, and Llama-3.2 with hidden-state extraction
and special boundary tokens.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Special tokens used as span boundary markers
SPAN_START_TOKEN = "<SPAN_START>"
SPAN_END_TOKEN = "<SPAN_END>"
LATENT_TOKEN = "<LATENT>"
SPECIAL_TOKENS = [SPAN_START_TOKEN, SPAN_END_TOKEN, LATENT_TOKEN]


class LatentReasoningModel(nn.Module):
    """Unified wrapper around HuggingFace causal-LM models.

    Handles tokenizer expansion, hidden-state extraction, and generation
    in a backend-agnostic way (GPT-2 / Qwen2.5 / Llama-3.2).

    Args:
        model_name: HuggingFace model identifier or local path.
        layer_ids: List of layer indices for hidden-state alignment.
                   Negative values are counted from the last layer.
        num_latent_tokens: Number of latent reasoning tokens.
        device: Target device (``"cuda"`` / ``"cpu"``).
        torch_dtype: Data type for model weights.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        layer_ids: Optional[List[int]] = None,
        num_latent_tokens: int = 3,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.layer_ids = layer_ids or [-1, -2]
        self.num_latent_tokens = num_latent_tokens
        self.device = device

        # --- Load tokenizer & model ---
        logger.info("Loading tokenizer from %s", model_name)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models require left-padding for correct generation
        self.tokenizer.padding_side = "left"

        logger.info("Loading model from %s", model_name)
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # --- Add special tokens ---
        self._add_special_tokens()

        # --- Learnable latent token embeddings ---
        self._init_latent_embeddings()

        # Move model to device
        self.model.to(device)
        # Move latent embeddings to device (they are a separate nn.Module)
        self.latent_embeddings.to(device)
        logger.info(
            "Model loaded: %s | params=%.1fM | device=%s | num_latent_tokens=%d",
            model_name,
            sum(p.numel() for p in self.model.parameters()) / 1e6,
            device,
            self.num_latent_tokens,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _add_special_tokens(self) -> None:
        """Add ``<SPAN_START>``, ``<SPAN_END>``, and ``<LATENT>`` to the
        tokenizer and resize model embeddings accordingly."""
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(
                "Added %d special tokens → vocab size %d",
                num_added,
                len(self.tokenizer),
            )

        # Convenient references
        self.span_start_token_id: int = self.tokenizer.convert_tokens_to_ids(
            SPAN_START_TOKEN
        )
        self.span_end_token_id: int = self.tokenizer.convert_tokens_to_ids(
            SPAN_END_TOKEN
        )
        self.latent_token_id: int = self.tokenizer.convert_tokens_to_ids(
            LATENT_TOKEN
        )

    def _init_latent_embeddings(self) -> None:
        """Initialize learnable latent token embeddings.

        Creates an ``nn.Embedding`` with *num_latent_tokens* entries,
        each of dimension *hidden_dim* (matching the model’s hidden size).
        Initialization uses N(0, 0.02) matching typical transformer init.
        """
        hidden_dim = self.model.config.hidden_size
        self.latent_embeddings = nn.Embedding(self.num_latent_tokens, hidden_dim)
        # Initialize with small normal values
        nn.init.normal_(self.latent_embeddings.weight, mean=0.0, std=0.02)
        logger.info(
            "Latent embeddings initialised: num_tokens=%d, dim=%d",
            self.num_latent_tokens,
            hidden_dim,
        )

    # ------------------------------------------------------------------
    # Core interfaces
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a forward pass and return logits + (optionally) hidden states.

        When *inputs_embeds* is provided, it takes precedence over
        *input_ids* (used for student forward with latent injection).

        Returns:
            Dictionary with keys ``"logits"``, ``"loss"`` (if *labels*
            provided), and ``"hidden_states"`` (tuple of layer tensors,
            each ``[B, L, D]``).
        """
        if inputs_embeds is not None:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )

        result: Dict[str, Any] = {"logits": outputs.logits}

        if outputs.loss is not None:
            result["loss"] = outputs.loss

        if output_hidden_states and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states

        return result

    def forward_with_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        latent_positions: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Student forward pass: inject learned latent embeddings.

        Replaces the token embeddings at *latent_positions* with the
        corresponding entries from ``self.latent_embeddings`` before
        running the transformer forward pass.

        Args:
            input_ids: Token ids ``[B, L]`` (with ``<LATENT>`` placeholders).
            attention_mask: Mask ``[B, L]``.
            labels: Optional labels ``[B, L]`` for generation loss.
            latent_positions: ``[B, K]`` positions of latent tokens per
                sample (K = num_latent_tokens).
            output_hidden_states: Whether to return all hidden states.

        Returns:
            Same dictionary as :meth:`forward`.
        """
        # Step 1: Get token embeddings from the model's embedding layer
        inputs_embeds = self.model.get_input_embeddings()(input_ids).clone()

        # Step 2: Inject learned latent embeddings at specified positions
        if latent_positions is not None:
            B, K = latent_positions.shape
            for k in range(K):
                # Each latent position k gets the k-th learned embedding
                latent_emb = self.latent_embeddings(
                    torch.tensor(k, device=input_ids.device)
                )  # [D]
                for b in range(B):
                    pos = latent_positions[b, k].item()
                    if pos > 0:  # 0 is padding
                        inputs_embeds[b, pos, :] = latent_emb

        # Step 3: Forward with the modified embeddings
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        latent_positions: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate with latent embedding injection.

        If *latent_positions* is provided (Stage 1 student mode),
        injects learned latent embeddings before generation.
        Otherwise falls back to standard generation.
        """
        if latent_positions is not None and self.num_latent_tokens > 0:
            # Inject learned latent embeddings at specified positions
            inputs_embeds = self.model.get_input_embeddings()(input_ids).clone()
            B, K = latent_positions.shape
            for k in range(min(K, self.num_latent_tokens)):
                latent_emb = self.latent_embeddings(
                    torch.tensor(k, device=input_ids.device)
                )
                for b in range(B):
                    pos = latent_positions[b, k].item()
                    if pos > 0:  # skip padding
                        inputs_embeds[b, pos, :] = latent_emb
            return self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Standard generation (Stage 0 / no latent tokens)
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

    @torch.no_grad()
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_ids: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Extract hidden states at specified layers.

        Args:
            input_ids: Token ids, shape ``[B, L]``.
            attention_mask: Optional mask, shape ``[B, L]``.
            layer_ids: Layer indices to extract.  Defaults to
                       ``self.layer_ids``.

        Returns:
            Mapping from *resolved* (positive) layer index to the
            hidden-state tensor of shape ``[B, L, D]``.
        """
        layer_ids = layer_ids or self.layer_ids
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hidden = outputs.hidden_states  # tuple of (num_layers+1,) tensors
        num_layers = len(all_hidden)

        extracted: Dict[int, torch.Tensor] = {}
        for lid in layer_ids:
            resolved = lid if lid >= 0 else num_layers + lid
            if 0 <= resolved < num_layers:
                extracted[resolved] = all_hidden[resolved]
            else:
                raise IndexError(
                    f"Layer index {lid} (resolved={resolved}) out of range "
                    f"[0, {num_layers})"
                )
        return extracted

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def get_num_layers(self) -> int:
        """Return the total number of transformer layers (excl. embedding)."""
        outputs = self.model(
            input_ids=torch.tensor([[0]], device=self.device),
            output_hidden_states=True,
        )
        return len(outputs.hidden_states) - 1  # exclude embedding layer

    def save_pretrained(self, save_dir: str) -> None:
        """Persist model and tokenizer to *save_dir*."""
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info("Saved model and tokenizer to %s", save_dir)

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str,
        layer_ids: Optional[List[int]] = None,
        device: str = "cuda",
    ) -> "LatentReasoningModel":
        """Load a previously saved model."""
        return cls(
            model_name=save_dir,
            layer_ids=layer_ids,
            device=device,
        )

    def freeze(self) -> None:
        """Freeze all parameters (useful for the teacher model)."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("All parameters frozen.")

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen.")
