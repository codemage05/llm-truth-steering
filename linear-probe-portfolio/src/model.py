"""
Model loading logic, Hugging Face gated model fallback handling,
and leak-free PyTorch forward hook registration for activation extraction.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import torch

DEFAULT_MODEL_NAME = "google/gemma-2b-it"
FALLBACK_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model_and_tokenizer(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    force_hf_native: bool = False,
) -> Tuple[Any, Any, int, int, bool, str]:
    """
    Loads model and tokenizer with automatic TransformerLens support and gated HF model fallback.

    Args:
        model_name: Requested Hugging Face model identifier (defaults to google/gemma-2b-it).
        device: 'cuda' or 'cpu'. Auto-detects if None.
        force_hf_native: If True, bypasses TransformerLens and forces native HF transformers.

    Returns:
        Tuple of (model, tokenizer, n_layers, hidden_dim, use_transformer_lens, selected_model_name)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    target_model = model_name or DEFAULT_MODEL_NAME

    # Check HF authentication token for gated models like google/gemma-2b-it
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if "gemma" in target_model.lower() and not hf_token:
        print(
            f"[HF Token Check] No HF_TOKEN found in environment. Automatically falling back "
            f"from gated model '{target_model}' to ungated model '{FALLBACK_MODEL_NAME}'."
        )
        target_model = FALLBACK_MODEL_NAME

    model = None
    tokenizer = None
    n_layers = None
    hidden_dim = None
    use_transformer_lens = False

    dtype = torch.float16 if device == "cuda" else torch.float32

    if not force_hf_native:
        try:
            from transformer_lens import HookedTransformer

            print(f"[TransformerLens] Attempting to load {target_model}...")
            model = HookedTransformer.from_pretrained(
                target_model,
                dtype=dtype,
                device=device,
            )
            tokenizer = model.tokenizer
            n_layers = model.cfg.n_layers
            hidden_dim = model.cfg.d_model
            use_transformer_lens = True
            print(f"[TransformerLens] Loaded {target_model} | layers={n_layers} | d_model={hidden_dim}")
            return model, tokenizer, n_layers, hidden_dim, use_transformer_lens, target_model

        except Exception as e:
            print(f"[TransformerLens] Load failed ({type(e).__name__}: {e})")
            if target_model != FALLBACK_MODEL_NAME and ("gated" in str(e).lower() or "401" in str(e) or "403" in str(e)):
                print(f"[Fallback] Falling back to {FALLBACK_MODEL_NAME}...")
                target_model = FALLBACK_MODEL_NAME

    # Native HF Transformers Fallback Path
    print(f"[HF Native] Loading {target_model} via native transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    except Exception as e:
        if target_model != FALLBACK_MODEL_NAME:
            print(f"[HF Native Auth Failure] Loading fallback model {FALLBACK_MODEL_NAME} instead...")
            target_model = FALLBACK_MODEL_NAME
            tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
        else:
            raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        target_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    n_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", None))
    hidden_dim = getattr(model.config, "hidden_size", getattr(model.config, "d_model", None))

    print(f"[HF Native] Loaded {target_model} | layers={n_layers} | d_model={hidden_dim}")
    return model, tokenizer, n_layers, hidden_dim, False, target_model


class ResidualStreamCapture:
    """
    Captures the LAST-token residual stream output of specified decoder
    layers during a single forward pass, then self-removes its hooks.

    Guarantees cleanup on exit (even when exceptions occur mid-forward pass)
    to prevent GPU memory leaks and stale hook accumulation.
    """

    def __init__(self, hf_model: Any, layers: List[int]):
        self.model = hf_model
        self.layers = layers
        self.activations: Dict[int, torch.Tensor] = {}
        self._handles: List[Any] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # Decoder layers in HF causal LMs return a tuple whose first element is hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            # Retain only the last token position (decision point before generation)
            self.activations[layer_idx] = hidden[:, -1, :].detach().to(torch.float32).cpu()

        return hook

    def __enter__(self):
        # Attribute path compatible with Llama/Gemma/Qwen decoder layer structures
        decoder_layers = getattr(
            self.model.model,
            "layers",
            getattr(self.model, "transformer", getattr(self.model, "layers", None)),
        )
        if decoder_layers is None and hasattr(self.model.model, "decoder"):
            decoder_layers = self.model.model.decoder.layers

        for layer_idx in self.layers:
            handle = decoder_layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False  # Do not suppress exceptions
