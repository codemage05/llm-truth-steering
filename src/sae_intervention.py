"""
Residual-error-preserving SAE steering (Option C, Phases 2-3).

An SAE reconstructs only part of the residual stream: x = x_hat + e. Replacing x
with SAE.decode(z_modified) throws away the error term e and damages fluency even
at zero steering strength. Every SAE intervention here therefore edits only the
difference between the modified and clean reconstructions (Marks et al. 2024,
"Sparse Feature Circuits"; Templeton et al. 2024):

    x_steered = x + (x_hat_modified - x_hat_clean)
              = x + sum_{i in S} (z~_i - z_i) * W_dec[i] / s

where s is the SAE's scalar input normalization.

Two intervention families:

1. ``sae_feature_clamp_hook``: zero-ablate (or negatively scale) deception latents
   and clamp honesty latents up to a target magnitude at every position.
2. ``compute_sae_steering_vector``: project the dense mass-mean direction onto the
   span of the selected SAE decoder atoms, giving a denoised direction that plugs
   into the existing norm-scaled hook (``src.intervention.steer_generation``) via
   ``make_sparse_layer_result``.

Edits are ``ResidualStreamEdit`` objects. They work as HF forward hooks
(``layer.register_forward_hook(edit)``) and as TransformerLens hooks
(``edit.tl_hook``), and they record how large the perturbation was relative to the
activation norm, so collateral drift can be compared across methods.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .probe import LayerResult
from .sae import SparseAutoencoder, TruthFeatureReport, resolve_device

ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Base edit + concrete edits
# ---------------------------------------------------------------------------

class ResidualStreamEdit:
    """
    Base class for residual stream edits applied at one decoder layer.

    Subclasses implement ``transform(x)`` on float32 hidden states of shape
    (batch, seq, d_model). The edit casts back to the model dtype and tracks the
    mean relative perturbation ||delta|| / ||x|| over all edited positions.
    """

    def __init__(self) -> None:
        self.reset_stats()

    def reset_stats(self) -> None:
        self._ratio_sum = 0.0
        self._norm_sum = 0.0
        self._count = 0

    @property
    def mean_relative_perturbation(self) -> float:
        return self._ratio_sum / self._count if self._count else 0.0

    @property
    def mean_perturbation_norm(self) -> float:
        return self._norm_sum / self._count if self._count else 0.0

    def transform(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    @torch.no_grad()
    def edit(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.float()
        new = self.transform(x)
        delta_norm = (new - x).norm(dim=-1)
        self._ratio_sum += float((delta_norm / x.norm(dim=-1).clamp_min(1e-6)).sum())
        self._norm_sum += float(delta_norm.sum())
        self._count += delta_norm.numel()
        return new.to(hidden.dtype)

    # HF forward hook signature
    def __call__(self, module: Any, inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        new_hidden = self.edit(hidden)
        return (new_hidden,) + tuple(output[1:]) if isinstance(output, tuple) else new_hidden

    # TransformerLens hook signature
    def tl_hook(self, resid: torch.Tensor, hook: Any = None) -> torch.Tensor:
        return self.edit(resid)


class DenseSteeringEdit(ResidualStreamEdit):
    """
    Constant-vector activation addition with the exact math from
    ``src.intervention.steer_generation``:

        v = -strength * avg_activation_norm * direction / 10

    ``direction`` points from truth to lie, so positive strength steers toward truth.
    """

    def __init__(self, direction: ArrayLike, avg_activation_norm: float, strength: float):
        super().__init__()
        d = torch.as_tensor(np.asarray(direction, dtype=np.float32))
        self.vector = -float(strength) * float(avg_activation_norm) * d / 10.0

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.vector.device != x.device:
            self.vector = self.vector.to(x.device)
        return x + self.vector


class SAEFeatureClampHook(ResidualStreamEdit):
    """
    Sparse feature intervention on selected SAE latents.

    - Deception latents: z~_i = deception_scale * z_i (0 = zero-ablation, <0 = invert).
    - Honesty latents:   z~_i = max(z_i, target_i) (clamp up, never suppress).

    With ``preserve_error=True`` (default) the edit is
    x + sum_i (z~_i - z_i) W_dec[i] / s, which keeps the SAE error term. With
    ``preserve_error=False`` the residual stream is replaced by SAE.decode(z~); this
    is included only as an ablation showing what discarding the error does.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        deception_features: Sequence[int],
        honesty_features: Sequence[int],
        deception_scale: float = 0.0,
        honesty_targets: Optional[ArrayLike] = None,
        preserve_error: bool = True,
    ):
        super().__init__()
        self.sae = sae.eval()
        self.dec_idx = torch.as_tensor(list(deception_features), dtype=torch.long)
        self.hon_idx = torch.as_tensor(list(honesty_features), dtype=torch.long)
        if set(self.dec_idx.tolist()) & set(self.hon_idx.tolist()):
            raise ValueError("deception_features and honesty_features must be disjoint.")
        self.deception_scale = float(deception_scale)
        if honesty_targets is None:
            honesty_targets = torch.zeros(len(self.hon_idx))
        self.hon_targets = torch.as_tensor(np.asarray(honesty_targets, dtype=np.float32)).reshape(-1)
        if self.hon_targets.numel() != self.hon_idx.numel():
            raise ValueError("honesty_targets must have one value per honesty feature.")
        self.preserve_error = preserve_error

    def _sync_device(self, device: torch.device) -> None:
        if self.sae.device != device:
            self.sae.to(device)
        if self.dec_idx.device != device:
            self.dec_idx, self.hon_idx = self.dec_idx.to(device), self.hon_idx.to(device)
            self.hon_targets = self.hon_targets.to(device)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self._sync_device(x.device)
        z = self.sae.encode(x)
        z_dec = z[..., self.dec_idx]
        z_hon = z[..., self.hon_idx]
        new_dec = self.deception_scale * z_dec
        new_hon = torch.maximum(z_hon, self.hon_targets)

        if self.preserve_error:
            W = self.sae.W_dec
            delta = (new_dec - z_dec) @ W[self.dec_idx] + (new_hon - z_hon) @ W[self.hon_idx]
            return x + delta / self.sae.input_scale

        z_mod = z.clone()
        z_mod[..., self.dec_idx] = new_dec
        z_mod[..., self.hon_idx] = new_hon
        return self.sae.decode(z_mod)


def _select_features(
    report: Optional[TruthFeatureReport],
    deception_features: Optional[Sequence[int]],
    honesty_features: Optional[Sequence[int]],
    k_deception: int,
    k_honesty: int,
) -> Tuple[List[int], List[int]]:
    if deception_features is None or honesty_features is None:
        if report is None:
            raise ValueError("Provide a TruthFeatureReport or explicit feature lists.")
    dec = list(deception_features) if deception_features is not None else report.top_deception_features[:k_deception]
    hon = list(honesty_features) if honesty_features is not None else report.top_honesty_features[:k_honesty]
    return [int(i) for i in dec], [int(i) for i in hon]


def sae_feature_clamp_hook(
    sae: SparseAutoencoder,
    report: Optional[TruthFeatureReport] = None,
    strength: float = 1.0,
    *,
    deception_features: Optional[Sequence[int]] = None,
    honesty_features: Optional[Sequence[int]] = None,
    k_deception: int = 5,
    k_honesty: int = 5,
    deception_scale: float = 0.0,
    reference: str = "norm",
    avg_activation_norm: Optional[float] = None,
    preserve_error: bool = True,
) -> SAEFeatureClampHook:
    """
    Build an error-preserving feature clamp/ablation hook.

    Args:
        sae: Trained ``SparseAutoencoder`` for the steered layer.
        report: ``TruthFeatureReport`` from ``discover_truth_features``; supplies the
            top-k feature indices (unless given explicitly) and firing magnitudes.
        strength: Honesty clamp multiplier. ``strength == 0`` returns an identity edit.
        deception_scale: Applied to deception latents whenever strength > 0
            (0 = zero-ablate, negative = push against the feature).
        reference: How honesty targets are scaled.
            ``"norm"``: target_i = strength * avg_activation_norm * s / (10 * sqrt(k_hon)).
                The total clamp is then on the same scale as dense steering at the same
                strength (strength * avg_norm / 10), which allows matched-norm comparisons.
            ``"firing"``: target_i = strength * E[z_i | honest prompt, z_i > 0].
        avg_activation_norm: Required for ``reference="norm"`` (``LayerResult.avg_activation_norm``).
    """
    dec, hon = _select_features(report, deception_features, honesty_features, k_deception, k_honesty)
    if strength == 0:
        return SAEFeatureClampHook(sae, [], [], deception_scale=1.0, preserve_error=preserve_error)

    if not hon:
        targets = np.zeros(0, dtype=np.float32)
    elif reference == "norm":
        if avg_activation_norm is None:
            raise ValueError('reference="norm" requires avg_activation_norm.')
        per_feature = strength * avg_activation_norm * float(sae.input_scale) / (10.0 * np.sqrt(len(hon)))
        targets = np.full(len(hon), per_feature, dtype=np.float32)
    elif reference == "firing":
        if report is None:
            raise ValueError('reference="firing" requires a TruthFeatureReport.')
        firing = report.mu_honest[hon] / np.maximum(report.freq_honest[hon], 1e-6)
        targets = (strength * firing).astype(np.float32)
    else:
        raise ValueError(f"reference must be 'norm' or 'firing', got {reference!r}.")

    return SAEFeatureClampHook(
        sae, dec, hon, deception_scale=deception_scale, honesty_targets=targets, preserve_error=preserve_error
    )


# ---------------------------------------------------------------------------
# Sparse subspace steering vector
# ---------------------------------------------------------------------------

@dataclass
class SparseSteeringVector:
    """Dense truth->lie direction projected onto the span of selected SAE decoder atoms."""
    direction: np.ndarray = field(repr=False)        # (d_model,) unit norm, truth -> lie
    feature_indices: List[int]
    coefficients: np.ndarray = field(repr=False)     # least-squares weight of each atom
    cosine_with_dense: float                         # = ||P d|| for unit d
    residual_fraction: float                         # ||(I - P) d||, the part removed as non-feature noise
    subspace_rank: int


def compute_sae_steering_vector(
    sae: SparseAutoencoder,
    mass_mean_direction: ArrayLike,
    report: Optional[TruthFeatureReport] = None,
    *,
    deception_features: Optional[Sequence[int]] = None,
    honesty_features: Optional[Sequence[int]] = None,
    k_deception: int = 5,
    k_honesty: int = 5,
    rcond: float = 1e-6,
) -> SparseSteeringVector:
    """
    Purify the dense steering direction by orthogonally projecting it onto
    span{W_dec[i] : i in S}, where S holds the selected deception and honesty latents.

    Components of the mass-mean vector outside this subspace (polysemantic or
    unrelated directions) are removed. The result is unit-normalised and keeps the
    truth->lie sign convention, so it can replace ``LayerResult.direction`` in the
    existing norm-scaled hook (see ``make_sparse_layer_result``).
    """
    dec, hon = _select_features(report, deception_features, honesty_features, k_deception, k_honesty)
    indices = dec + hon
    if not indices:
        raise ValueError("No SAE features selected for the steering subspace.")

    d = np.asarray(mass_mean_direction, dtype=np.float64).reshape(-1)
    d = d / (np.linalg.norm(d) + 1e-12)
    atoms = sae.feature_directions[indices].detach().cpu().double().numpy()  # (k, d_model)

    U, svals, _ = np.linalg.svd(atoms.T, full_matrices=False)
    rank = int((svals > rcond * svals[0]).sum())
    basis = U[:, :rank]
    proj = basis @ (basis.T @ d)
    proj_norm = float(np.linalg.norm(proj))
    if proj_norm < 1e-8:
        raise ValueError("Mass-mean direction is orthogonal to the selected SAE feature subspace.")

    coefficients, *_ = np.linalg.lstsq(atoms.T, d, rcond=None)
    return SparseSteeringVector(
        direction=(proj / proj_norm).astype(np.float32),
        feature_indices=indices,
        coefficients=coefficients.astype(np.float32),
        cosine_with_dense=proj_norm,
        residual_fraction=float(np.sqrt(max(0.0, 1.0 - proj_norm ** 2))),
        subspace_rank=rank,
    )


def make_sparse_layer_result(layer_result: LayerResult, sparse_vector: Union[SparseSteeringVector, ArrayLike]) -> LayerResult:
    """Copy of ``layer_result`` whose ``direction`` is the purified sparse direction."""
    direction = sparse_vector.direction if isinstance(sparse_vector, SparseSteeringVector) else sparse_vector
    return replace(layer_result, direction=np.asarray(direction, dtype=np.float32))


# ---------------------------------------------------------------------------
# Hook registration + generation
# ---------------------------------------------------------------------------

def _decoder_layers(model: Any) -> Any:
    """Decoder block list for Llama/Gemma/Qwen-style HF models (mirrors src.intervention)."""
    decoder_layers = getattr(
        model.model,
        "layers",
        getattr(model, "transformer", getattr(model, "layers", None)),
    )
    if decoder_layers is None and hasattr(model.model, "decoder"):
        decoder_layers = model.model.decoder.layers
    return decoder_layers


@contextlib.contextmanager
def residual_edit_hook(
    model: Any, layer: int, edit: Optional[ResidualStreamEdit], use_transformer_lens: bool = False
) -> Iterator[None]:
    """Attach ``edit`` to the output of decoder block ``layer``; always removed on exit."""
    if edit is None:
        yield
        return
    if use_transformer_lens:
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", edit.tl_hook)]):
            yield
        return
    handle = _decoder_layers(model)[layer].register_forward_hook(edit)
    try:
        yield
    finally:
        handle.remove()


def _infer_device(model: Any) -> str:
    return str(resolve_device(None, model))


def steer_generation_with_edit(
    model: Any,
    tokenizer: Any,
    prompt_question: str,
    layer: int,
    edit: Optional[ResidualStreamEdit],
    max_new_tokens: int = 40,
    device: Optional[str] = None,
    use_transformer_lens: bool = True,
) -> str:
    """
    Single-prompt generation under an arbitrary ``ResidualStreamEdit``. Prompt
    formatting and return format (full decoded text) match ``steer_generation``.
    """
    device = device or _infer_device(model)
    chat = [{"role": "user", "content": prompt_question}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    if use_transformer_lens:
        tokens = model.to_tokens(text)
        with residual_edit_hook(model, layer, edit, use_transformer_lens=True):
            out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
        return model.to_string(out[0])

    inputs = tokenizer(text, return_tensors="pt").to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with residual_edit_hook(model, layer, edit), torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def _stop_token_ids(model: Any, tokenizer: Any) -> set:
    ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gen_eos, (list, tuple)):
        ids.update(gen_eos)
    elif gen_eos is not None:
        ids.add(gen_eos)
    return {i for i in ids if i is not None}


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompt_texts: Sequence[str],
    layer: int,
    edit: Optional[ResidualStreamEdit] = None,
    max_new_tokens: int = 24,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> List[Tuple[List[int], str]]:
    """
    Greedy batched generation (HF models) for already chat-formatted prompts, with
    ``edit`` applied at ``layer``. Prompts are left-padded and attention-masked.

    Returns one (generated_token_ids, decoded_response) pair per prompt, truncated
    at the first stop token.
    """
    if hasattr(model, "cfg") and hasattr(model, "to_tokens"):
        raise NotImplementedError("generate_batch supports HF models; use steer_generation_with_edit for TransformerLens.")
    device = device or _infer_device(model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stop_ids = _stop_token_ids(model, tokenizer)
    results: List[Tuple[List[int], str]] = []

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompt_texts), batch_size):
            batch = list(prompt_texts[start:start + batch_size])
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            with residual_edit_hook(model, layer, edit), torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
                    temperature=None, top_p=None, top_k=None,
                )
            for row in out[:, enc["input_ids"].shape[1]:].tolist():
                cut = next((i for i, t in enumerate(row) if t in stop_ids), len(row))
                ids = row[:cut]
                results.append((ids, tokenizer.decode(ids, skip_special_tokens=True).strip()))
    finally:
        tokenizer.padding_side = original_side
    return results
