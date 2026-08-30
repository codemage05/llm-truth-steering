"""
Causal intervention / Activation Addition (Representation Engineering) logic.
Subtracts/adds the truth/lie activation direction during model generation.
"""

from typing import Any, Optional
import torch
from .probe import LayerResult


def steer_generation(
    model: Any,
    tokenizer: Any,
    prompt_question: str,
    layer_result: LayerResult,
    strength: float = 6.0,
    max_new_tokens: int = 40,
    device: Optional[str] = None,
    use_transformer_lens: bool = True,
) -> str:
    """
    Generate a completion for a prompt while modifying the residual stream at target layer.

    Preserves exact Steering Math:
    steering_vector = -strength * layer_result.avg_activation_norm * layer_result.direction / 10.0

    This scales steering relative to the layer's average activation norm so offsets stay
    proportional regardless of depth or model dimension.
    """
    if device is None:
        if hasattr(model, "cfg") and hasattr(model.cfg, "device"):
            device = str(model.cfg.device)
        elif hasattr(model, "device"):
            device = str(model.device)
        else:
            device = str(next(model.parameters()).device)

    steering_vector_np = -strength * layer_result.avg_activation_norm * layer_result.direction / 10.0
    layer = layer_result.layer

    torch_dtype = torch.float16 if "cuda" in str(device) else torch.float32
    if use_transformer_lens:
        steering_vector = torch.tensor(steering_vector_np, dtype=torch_dtype, device=device)
        hook_name = f"blocks.{layer}.hook_resid_post"

        def add_steering(resid, hook):
            return resid + steering_vector

        chat = [{"role": "user", "content": prompt_question}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        tokens = model.to_tokens(text)

        with model.hooks(fwd_hooks=[(hook_name, add_steering)]):
            output_tokens = model.generate(
                tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False
            )
        return model.to_string(output_tokens[0])

    else:
        torch_dtype = torch.float16 if "cuda" in str(device) else torch.float32
        steering_vector = torch.tensor(steering_vector_np, dtype=torch_dtype, device=device)

        def steering_hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + steering_vector
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        chat = [{"role": "user", "content": prompt_question}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        decoder_layers = getattr(
            model.model,
            "layers",
            getattr(model, "transformer", getattr(model, "layers", None)),
        )
        if decoder_layers is None and hasattr(model.model, "decoder"):
            decoder_layers = model.model.decoder.layers

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        handle = decoder_layers[layer].register_forward_hook(steering_hook)
        try:
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
        finally:
            handle.remove()  # Guaranteed cleanup even if generate() raises an exception

        return tokenizer.decode(out_ids[0], skip_special_tokens=True)
