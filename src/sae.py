"""
Sparse Autoencoder (SAE) module for Option C: SAE Hybrid Steering.

Dense linear steering (src/intervention.py) adds a single mass-mean direction to
the residual stream. Because the residual stream is in superposition, that
direction is a blend of many features, so steering along it also degrades
unrelated behaviour (collateral coherence loss). This module decomposes the
residual stream into a sparse, overcomplete dictionary of (approximately)
monosemantic latent features so that later phases can steer only the specific
features that encode deception vs. honesty.

Contents
--------
- ``SparseAutoencoder``: native PyTorch SAE with ReLU / Top-K / JumpReLU
  sparsity and unit-norm decoder dictionary atoms.
- ``train_sae``: deterministic trainer (reconstruction MSE + L1 sparsity).
- ``load_or_train_sae``: loads a pretrained sae-lens SAE (Gemma-2B) when
  explicitly requested and available, otherwise trains a native SAE.
- ``extract_token_activations``: token-level residual stream extraction (SAE
  training data), reusing ``ResidualStreamCapture`` from ``src/model.py``.
- ``discover_truth_features``: differential activation analysis over the
  contrastive Honest vs. Deceptive prompts from ``src/data.py``.

Conventions (shared with the rest of the repo)
----------------------------------------------
- Activations are float32 arrays/tensors of shape ``(N, d_model)``, exactly as
  produced by ``src.data.extract_activations`` (``X_by_layer[layer]``).
- Labels follow ``src.data.LABEL_MAP``: 0 = truth/honest, 1 = lie/deceptive.
- Layer ``l`` refers to the output of decoder block ``l`` (TransformerLens
  ``blocks.{l}.hook_resid_post``), matching ``ResidualStreamCapture``.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import LABEL_MAP
from .model import ResidualStreamCapture

ArrayLike = Union[np.ndarray, torch.Tensor]
ActivationsInput = Union[ArrayLike, Dict[int, ArrayLike]]

HONEST_LABEL: int = LABEL_MAP["truth"]
DECEPTIVE_LABEL: int = LABEL_MAP["lie"]

_VALID_ACTIVATIONS = ("relu", "topk", "jumprelu")
_VALID_NORMALIZATIONS = ("std", "frequency", "none")


# ---------------------------------------------------------------------------
# Device / tensor utilities
# ---------------------------------------------------------------------------

def resolve_device(device: Optional[Union[str, torch.device]] = None, model: Any = None) -> torch.device:
    """
    Resolve the compute device.

    Priority: explicit ``device`` -> device of ``model`` (TransformerLens ``cfg.device``
    or first HF parameter) -> cuda -> mps -> cpu.
    """
    if device is not None:
        return torch.device(device)

    if model is not None:
        cfg = getattr(model, "cfg", None)
        if cfg is not None and getattr(cfg, "device", None) is not None:
            return torch.device(cfg.device)
        try:
            param_device = next(model.parameters()).device
            if param_device.type != "meta":
                return param_device
        except (AttributeError, StopIteration, TypeError):
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_layer(activations: ActivationsInput, layer: Optional[int]) -> ArrayLike:
    """Accept either a single activation matrix or an ``X_by_layer`` dict."""
    if isinstance(activations, dict):
        if layer is None:
            raise ValueError("`layer` must be given when passing an X_by_layer dict.")
        if layer not in activations:
            raise KeyError(f"Layer {layer} not in activations dict (available: {sorted(activations)}).")
        return activations[layer]
    return activations


def _as_float_tensor(x: ArrayLike) -> torch.Tensor:
    """Convert numpy/torch activations to a contiguous float32 CPU tensor of shape (N, d_model)."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    elif isinstance(x, torch.Tensor):
        t = x.detach().to(device="cpu", dtype=torch.float32)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x).__name__}.")

    if t.ndim == 3:  # (batch, seq, d_model) -> (batch*seq, d_model)
        t = t.reshape(-1, t.shape[-1])
    if t.ndim != 2:
        raise ValueError(f"Activations must have shape (N, d_model); got {tuple(t.shape)}.")
    if t.shape[0] == 0:
        raise ValueError("Activations tensor is empty.")
    if not torch.isfinite(t).all():
        raise ValueError("Activations contain NaN/Inf values.")
    return t.contiguous()


def _infer_d_model(model: Any) -> Optional[int]:
    if model is None:
        return None
    cfg = getattr(model, "cfg", None)
    if cfg is not None and getattr(cfg, "d_model", None) is not None:
        return int(cfg.d_model)
    config = getattr(model, "config", None)
    if config is not None:
        d = getattr(config, "hidden_size", getattr(config, "d_model", None))
        if d is not None:
            return int(d)
    return None


def _infer_model_name(model: Any) -> str:
    if model is None:
        return ""
    cfg = getattr(model, "cfg", None)
    if cfg is not None and getattr(cfg, "model_name", None):
        return str(cfg.model_name)
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "_name_or_path", None):
        return str(config._name_or_path)
    return str(getattr(model, "name_or_path", ""))


# ---------------------------------------------------------------------------
# Sparse Autoencoder
# ---------------------------------------------------------------------------

@dataclass
class SAEConfig:
    d_model: int
    d_sae: int
    activation: str = "relu"
    k: Optional[int] = None
    apply_b_dec_to_input: bool = True


class SAEOutput(NamedTuple):
    x_hat: torch.Tensor     # (N, d_model) reconstruction in original activation units
    z: torch.Tensor         # (N, d_sae) sparse latent activations
    pre_acts: torch.Tensor  # (N, d_sae) encoder pre-activations


@dataclass
class SAEMetrics:
    """Reconstruction / sparsity diagnostics for an SAE on a set of activations."""
    mse: float                  # per-element MSE in normalized units (E||x|| = sqrt(d_model))
    explained_variance: float   # 1 - SSE / SST in original units
    mean_l0: float              # mean number of active latents per token
    dead_features: int          # latents that never fire on these activations
    dead_fraction: float
    n_samples: int
    train_time_s: float = 0.0
    loss_history: List[float] = field(default_factory=list, repr=False)


class SparseAutoencoder(nn.Module):
    """
    Overcomplete sparse autoencoder over residual stream activations.

        x_n   = s * x                                   (scalar input normalization)
        z     = sigma( (x_n - b_dec) W_enc + b_enc )     W_enc in R^{d_model x d_sae}
        x_hat = ( z W_dec + b_dec ) / s                 W_dec in R^{d_sae x d_model}

    ``sigma`` is ReLU, Top-K (keep the k largest pre-activations, then ReLU), or
    JumpReLU (for converted pretrained SAEs). Each latent's dictionary atom
    ``W_dec[i]`` (a column of the d_model x d_sae dictionary matrix ``W_dec^T``) is
    constrained to unit L2 norm, so latent magnitudes are comparable and the L1
    penalty cannot be gamed by shrinking z while growing the decoder.

    ``s = sqrt(d_model) / E||x||`` makes hyperparameters (l1_coeff, lr) transfer
    across models whose residual stream norms differ by orders of magnitude
    (e.g. Gemma-2B vs Qwen2.5-0.5B). ``encode``/``decode`` accept and return
    activations in the model's original units; normalization is internal.
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        activation: str = "relu",
        k: Optional[int] = None,
        apply_b_dec_to_input: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(f"activation must be one of {_VALID_ACTIVATIONS}, got {activation!r}.")
        if activation == "topk" and (k is None or not 0 < k <= d_sae):
            raise ValueError(f"Top-K activation requires 0 < k <= d_sae ({d_sae}); got k={k}.")

        self.cfg = SAEConfig(
            d_model=int(d_model),
            d_sae=int(d_sae),
            activation=activation,
            k=int(k) if k is not None else None,
            apply_b_dec_to_input=apply_b_dec_to_input,
        )

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.register_buffer("threshold", torch.zeros(d_sae))  # used only by JumpReLU

        self.training_stats: Optional[SAEMetrics] = None
        self.source: str = "native"
        self.reset_parameters(seed)

    # -- properties --------------------------------------------------------

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    @property
    def d_sae(self) -> int:
        return self.cfg.d_sae

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    @property
    def feature_directions(self) -> torch.Tensor:
        """Unit-norm dictionary atoms in model space, shape (d_sae, d_model)."""
        return F.normalize(self.W_dec.detach(), dim=-1)

    def feature_direction(self, idx: int) -> torch.Tensor:
        """Unit-norm residual stream direction written by latent ``idx`` (d_model,)."""
        return self.feature_directions[idx]

    # -- initialization / constraints -------------------------------------

    @torch.no_grad()
    def reset_parameters(self, seed: int = 0) -> None:
        """
        Deterministic tied initialization: random unit-norm decoder atoms and
        W_enc = c * W_dec^T, with c = 2 * d_model / d_sae chosen so the initial
        ReLU reconstruction of an isotropic input is approximately the identity.
        """
        gen = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(self.d_sae, self.d_model, generator=gen)
        W = F.normalize(W, dim=-1)
        self.W_dec.copy_(W)
        self.W_enc.copy_(W.T * (2.0 * self.d_model / self.d_sae))
        self.b_enc.zero_()
        self.b_dec.zero_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Project decoder atoms back onto the unit sphere."""
        self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def remove_parallel_decoder_grad_(self) -> None:
        """Remove the gradient component that would change decoder atom norms."""
        if self.W_dec.grad is None:
            return
        W_unit = F.normalize(self.W_dec, dim=-1)
        parallel = (self.W_dec.grad * W_unit).sum(dim=-1, keepdim=True)
        self.W_dec.grad.sub_(parallel * W_unit)

    # -- forward -----------------------------------------------------------

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        x_n = x.to(self.W_enc.dtype) * self.input_scale
        if self.cfg.apply_b_dec_to_input:
            x_n = x_n - self.b_dec
        return x_n @ self.W_enc + self.b_enc

    def activation_fn(self, pre_acts: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation == "relu":
            return F.relu(pre_acts)
        if self.cfg.activation == "topk":
            values, indices = pre_acts.topk(self.cfg.k, dim=-1)
            return torch.zeros_like(pre_acts).scatter(-1, indices, F.relu(values))
        # JumpReLU: z = pre * H(pre - theta)
        return pre_acts * (pre_acts > self.threshold).to(pre_acts.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Residual stream activations (..., d_model) -> sparse latents (..., d_sae)."""
        return self.activation_fn(self.encode_pre(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Sparse latents (..., d_sae) -> reconstruction in original units (..., d_model)."""
        return (z @ self.W_dec + self.b_dec) / self.input_scale

    def forward(self, x: torch.Tensor) -> SAEOutput:
        pre_acts = self.encode_pre(x)
        z = self.activation_fn(pre_acts)
        return SAEOutput(x_hat=self.decode(z), z=z, pre_acts=pre_acts)

    def compute_loss(self, x: torch.Tensor, output: SAEOutput, l1_coeff: float) -> Dict[str, torch.Tensor]:
        """Reconstruction MSE (normalized units, per element) + l1_coeff * ||z||_1 (per token)."""
        mse = ((output.x_hat - x) * self.input_scale).pow(2).mean()
        l1 = output.z.abs().sum(dim=-1).mean()
        return {"loss": mse + l1_coeff * l1, "mse": mse, "l1": l1}

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({"cfg": asdict(self.cfg), "state_dict": self.state_dict(), "source": self.source}, path)

    @classmethod
    def load(cls, path: str, device: Optional[Union[str, torch.device]] = None) -> "SparseAutoencoder":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        sae = cls(**ckpt["cfg"])
        sae.load_state_dict(ckpt["state_dict"])
        sae.source = ckpt.get("source", "native")
        return sae.to(resolve_device(device)).eval()

    # -- sae-lens interop --------------------------------------------------

    @classmethod
    @torch.no_grad()
    def from_sae_lens(
        cls,
        sae_lens_sae: Any,
        calibration_activations: Optional[torch.Tensor] = None,
    ) -> "SparseAutoencoder":
        """
        Convert a loaded ``sae_lens.SAE`` into a native ``SparseAutoencoder`` so the
        rest of the pipeline uses one interface regardless of SAE origin.
        Supports standard (ReLU), Top-K and JumpReLU architectures.
        """
        cfg = sae_lens_sae.cfg
        d_in = int(getattr(cfg, "d_in"))
        d_sae = int(getattr(cfg, "d_sae"))

        arch = getattr(cfg, "architecture", "standard")
        arch = str(arch() if callable(arch) else arch).lower()
        act_name = str(getattr(cfg, "activation_fn_str", getattr(cfg, "activation_fn", "relu"))).lower()

        k = None
        if "jumprelu" in arch:
            activation = "jumprelu"
        elif "topk" in arch or "topk" in act_name:
            activation = "topk"
            k = getattr(cfg, "k", None) or (getattr(cfg, "activation_fn_kwargs", None) or {}).get("k")
            if k is None:
                raise ValueError("Top-K sae-lens SAE has no `k` in its config.")
        else:
            activation = "relu"

        sae = cls(
            d_model=d_in,
            d_sae=d_sae,
            activation=activation,
            k=k,
            apply_b_dec_to_input=bool(getattr(cfg, "apply_b_dec_to_input", True)),
        )
        sae.W_enc.copy_(sae_lens_sae.W_enc.detach().float().cpu())
        sae.b_enc.copy_(sae_lens_sae.b_enc.detach().float().cpu())
        sae.W_dec.copy_(sae_lens_sae.W_dec.detach().float().cpu())
        sae.b_dec.copy_(sae_lens_sae.b_dec.detach().float().cpu())
        if activation == "jumprelu":
            sae.threshold.copy_(sae_lens_sae.threshold.detach().float().cpu())

        norm_mode = str(getattr(cfg, "normalize_activations", "none") or "none")
        if norm_mode == "expected_average_only_in":
            if calibration_activations is None:
                raise ValueError("This SAE expects input normalization; pass calibration_activations.")
            mean_norm = calibration_activations.norm(dim=-1).mean().clamp_min(1e-8)
            sae.input_scale.fill_(math.sqrt(d_in) / float(mean_norm))
        elif norm_mode not in ("none", "None"):
            warnings.warn(f"sae-lens normalize_activations={norm_mode!r} is not supported; using scale=1.")
        return sae


# ---------------------------------------------------------------------------
# Evaluation & training
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_batched(
    sae: SparseAutoencoder,
    X: torch.Tensor,
    batch_size: int = 1024,
    return_recon: bool = False,
):
    """Encode (and optionally reconstruct) activations in batches; returns CPU tensors."""
    zs, recons = [], []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size].to(sae.device)
        out = sae(xb)
        zs.append(out.z.cpu())
        if return_recon:
            recons.append(out.x_hat.cpu())
    Z = torch.cat(zs, dim=0)
    return (Z, torch.cat(recons, dim=0)) if return_recon else Z


@torch.no_grad()
def evaluate_sae(sae: SparseAutoencoder, activations: ActivationsInput, layer: Optional[int] = None,
                 batch_size: int = 1024) -> SAEMetrics:
    """Compute reconstruction quality and sparsity of ``sae`` on ``activations``."""
    X = _as_float_tensor(_select_layer(activations, layer))
    was_training = sae.training
    sae.eval()
    Z, X_hat = _encode_batched(sae, X, batch_size, return_recon=True)
    sae.train(was_training)

    scale = float(sae.input_scale)
    sse = (X - X_hat).pow(2).sum()
    sst = (X - X.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-12)
    fires = (Z > 0)
    dead = int((~fires.any(dim=0)).sum())
    return SAEMetrics(
        mse=float(((X - X_hat) * scale).pow(2).mean()),
        explained_variance=float(1.0 - sse / sst),
        mean_l0=float(fires.sum(dim=-1).float().mean()),
        dead_features=dead,
        dead_fraction=dead / sae.d_sae,
        n_samples=int(X.shape[0]),
    )


def train_sae(
    activations: ActivationsInput,
    expansion_factor: int = 4,
    l1_coeff: float = 1e-3,
    epochs: int = 200,
    lr: float = 1e-3,
    *,
    layer: Optional[int] = None,
    batch_size: int = 256,
    activation: str = "relu",
    k: Optional[int] = None,
    l1_warmup_frac: float = 0.05,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> SparseAutoencoder:
    """
    Deterministically train a native ``SparseAutoencoder`` on residual stream activations.

    Loss = MSE(x_hat, x) + l1_coeff * ||z||_1, with a linear L1 warm-up (reduces
    dead latents), unit-norm decoder projection after every step, and gradient
    clipping. Initialization and minibatch order come from a seeded CPU generator,
    so runs are reproducible on CPU (and on GPU up to kernel-level nondeterminism).
    """
    X = _as_float_tensor(_select_layer(activations, layer))
    n_samples, d_model = X.shape
    d_sae = int(d_model * expansion_factor)
    dev = resolve_device(device)

    sae = SparseAutoencoder(d_model, d_sae, activation=activation, k=k, seed=seed)
    with torch.no_grad():
        mean_norm = X.norm(dim=-1).mean().clamp_min(1e-8)
        sae.input_scale.fill_(math.sqrt(d_model) / float(mean_norm))
        sae.b_dec.copy_((X * sae.input_scale).mean(dim=0))  # data-centred decoder bias

    sae.to(dev).train()
    X_dev = X.to(dev)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    bs = max(1, min(batch_size, n_samples))
    steps_per_epoch = math.ceil(n_samples / bs)
    warmup_steps = max(1, int(l1_warmup_frac * epochs * steps_per_epoch))
    log_every = max(1, epochs // 5)
    history: List[float] = []
    step = 0
    t0 = time.perf_counter()

    if verbose:
        print(
            f"[SAE] Training native SAE | N={n_samples} | d_model={d_model} -> d_sae={d_sae} "
            f"| act={activation}{f'(k={k})' if activation == 'topk' else ''} | l1={l1_coeff} "
            f"| epochs={epochs} | lr={lr} | device={dev}"
        )

    for epoch in range(epochs):
        perm = torch.randperm(n_samples, generator=gen).to(dev)
        epoch_loss = 0.0
        for b in range(steps_per_epoch):
            xb = X_dev[perm[b * bs:(b + 1) * bs]]
            coeff = l1_coeff * min(1.0, (step + 1) / warmup_steps)

            out = sae(xb)
            losses = sae.compute_loss(xb, out, coeff)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            sae.remove_parallel_decoder_grad_()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            sae.normalize_decoder_()

            epoch_loss += losses["loss"].detach().item() * xb.shape[0]
            step += 1

        history.append(epoch_loss / n_samples)
        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            with torch.no_grad():
                l0 = float((out.z > 0).sum(dim=-1).float().mean())
            print(
                f"  epoch {epoch + 1:>4}/{epochs} | loss={history[-1]:.4f} "
                f"| mse={losses['mse'].detach().item():.4f} | L0(batch)={l0:.1f}"
            )

    sae.eval()
    metrics = evaluate_sae(sae, X)
    metrics.train_time_s = time.perf_counter() - t0
    metrics.loss_history = history
    sae.training_stats = metrics
    sae.source = "native"

    if verbose:
        print(
            f"[SAE] Done in {metrics.train_time_s:.1f}s | explained var={metrics.explained_variance:.3f} "
            f"| mean L0={metrics.mean_l0:.1f}/{d_sae} | dead={metrics.dead_features} "
            f"({100 * metrics.dead_fraction:.1f}%)"
        )
    return sae


def _load_sae_lens_sae(release: str, sae_id: str, device: torch.device) -> Any:
    """Load an SAE via sae-lens, handling both <6.0 (tuple) and >=6.0 (object) return types."""
    from sae_lens import SAE  # optional dependency

    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device))
    return loaded[0] if isinstance(loaded, tuple) else loaded


def load_or_train_sae(
    model: Any,
    activations: ActivationsInput,
    layer: int,
    expansion_factor: int = 4,
    l1_coeff: float = 1e-3,
    epochs: int = 200,
    lr: float = 1e-3,
    *,
    pretrained_release: Optional[str] = None,
    pretrained_sae_id: Optional[str] = None,
    activation: str = "relu",
    k: Optional[int] = None,
    batch_size: int = 256,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
    min_pretrained_explained_variance: float = 0.5,
    verbose: bool = True,
) -> SparseAutoencoder:
    """
    Return an SAE for residual stream ``layer`` of ``model``.

    Pretrained path (opt-in): when ``pretrained_release`` is given (e.g.
    ``"gemma-2b-it-res-jb"`` with sae_id ``"blocks.12.hook_resid_post"``), ``sae_lens``
    is installed, the loaded model is Gemma and the SAE width matches the
    activations, the pretrained SAE is loaded and converted to a native
    ``SparseAutoencoder``. Its explained variance on ``activations`` is checked,
    because SAEs trained on TransformerLens activations can mismatch HF-native
    activations (TransformerLens centres writing weights by default).

    Native path (default / fallback): trains a ``SparseAutoencoder`` on
    ``activations`` with ``train_sae``. This works for any model (Gemma-2B-it or
    the Qwen2.5-0.5B-Instruct fallback). For a well-conditioned dictionary, prefer
    token-level activations from ``extract_token_activations`` over the 36
    last-token vectors from ``extract_activations``.

    Args:
        model: Loaded model from ``load_model_and_tokenizer`` (TransformerLens or HF);
            may be None, in which case device/width are inferred from ``activations``.
        activations: (N, d_model) array/tensor, or an ``X_by_layer`` dict.
        layer: Residual stream layer index the activations come from.

    Returns:
        ``SparseAutoencoder`` in eval mode with ``.source`` and ``.training_stats`` set.
    """
    X = _as_float_tensor(_select_layer(activations, layer))
    model_d = _infer_d_model(model)
    if model_d is not None and model_d != X.shape[1]:
        raise ValueError(f"Activation width {X.shape[1]} != model d_model {model_d}.")
    dev = resolve_device(device, model)

    if pretrained_release is not None:
        sae_id = pretrained_sae_id or f"blocks.{layer}.hook_resid_post"
        model_name = _infer_model_name(model)
        reason = None
        if model is not None and "gemma" not in model_name.lower():
            reason = (f"loaded model is '{model_name}' (likely the ungated fallback), but "
                      f"pretrained release '{pretrained_release}' targets Gemma")
        else:
            try:
                lens_sae = _load_sae_lens_sae(pretrained_release, sae_id, dev)
                sae = SparseAutoencoder.from_sae_lens(lens_sae, calibration_activations=X)
                del lens_sae
                if sae.d_model != X.shape[1]:
                    reason = f"SAE width {sae.d_model} != activation width {X.shape[1]}"
                else:
                    sae = sae.to(dev).eval()
                    sae.source = f"sae_lens:{pretrained_release}/{sae_id}"
                    sae.training_stats = evaluate_sae(sae, X)
                    ev = sae.training_stats.explained_variance
                    if verbose:
                        print(
                            f"[SAE] Loaded pretrained {sae.source} | d_sae={sae.d_sae} | explained var={ev:.3f} "
                            f"| mean L0={sae.training_stats.mean_l0:.1f}"
                        )
                    if ev < min_pretrained_explained_variance:
                        warnings.warn(
                            f"Pretrained SAE explains only {ev:.2f} of activation variance; activations may "
                            f"be preprocessed differently from the SAE's training data (TL vs HF-native)."
                        )
                    return sae
            except ImportError:
                reason = "sae_lens is not installed (pip install sae-lens)"
            except Exception as e:  # network / unknown release / unsupported architecture
                reason = f"{type(e).__name__}: {e}"

        warnings.warn(f"[SAE] Pretrained SAE unavailable ({reason}); training native SAE instead.")

    return train_sae(
        X,
        expansion_factor=expansion_factor,
        l1_coeff=l1_coeff,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        activation=activation,
        k=k,
        seed=seed,
        device=dev,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Token-level activation extraction (SAE training data)
# ---------------------------------------------------------------------------

class FullSequenceResidualCapture(ResidualStreamCapture):
    """
    ``ResidualStreamCapture`` variant that keeps ALL token positions of the
    (batch-size-1) forward pass: ``activations[layer]`` has shape (seq_len, d_model).
    Inherits leak-free hook registration/removal from the parent class.
    """

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden[0].detach().to(torch.float32).cpu()

        return hook


def extract_token_activations(
    model: Any,
    tokenizer: Any,
    records: List[dict],
    layer: int,
    device: Optional[str] = None,
    use_transformer_lens: bool = True,
    skip_first_n_tokens: int = 1,
    norm_outlier_factor: Optional[float] = 10.0,
    dedup_shared_prefix: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Collect residual stream activations at every token position of every prompt
    (as built by ``src.data.build_dataset``) for SAE training.

    Tokenization mirrors ``src.data.extract_activations`` so the SAE sees the same
    distribution as the last-token activations used for discovery.

    Args:
        skip_first_n_tokens: Drop the first positions (BOS / chat-start tokens act as
            attention sinks with norms 10-100x larger than other tokens).
        norm_outlier_factor: Additionally drop any token whose norm exceeds this
            multiple of the median norm (residual "massive activations"). None disables.
        dedup_shared_prefix: Keep each causal prefix only once. A position's activation
            depends only on tokens[:t+1], so shared chat-template / system-prompt
            prefixes would otherwise be duplicated once per prompt.

    Returns:
        float32 array of shape (total_tokens, d_model).
    """
    dev = str(resolve_device(device, model))
    chunks: List[np.ndarray] = []
    hook_name = f"blocks.{layer}.hook_resid_post"
    seen_prefixes: set = set()
    n_duplicates = 0

    for i, rec in enumerate(records):
        if use_transformer_lens:
            tokens = model.to_tokens(rec["prompt"])
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == hook_name)
            acts = cache[hook_name][0].detach().float().cpu()
            token_ids = tokens[0].tolist()
            del cache
        else:
            inputs = tokenizer(rec["prompt"], return_tensors="pt").to(dev)
            with torch.no_grad(), FullSequenceResidualCapture(model, [layer]) as capture:
                model(**inputs)
            acts = capture.activations[layer]
            token_ids = inputs["input_ids"][0].tolist()
            del inputs

        keep_positions = list(range(skip_first_n_tokens, acts.shape[0]))
        if dedup_shared_prefix:
            unique_positions = []
            for t in keep_positions:
                key = tuple(token_ids[: t + 1])
                if key in seen_prefixes:
                    n_duplicates += 1
                else:
                    seen_prefixes.add(key)
                    unique_positions.append(t)
            keep_positions = unique_positions

        chunks.append(acts[keep_positions].numpy())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose and ((i + 1) % 10 == 0 or (i + 1) == len(records)):
            print(f"  extracted tokens for {i + 1}/{len(records)} prompts")

    X = np.concatenate(chunks, axis=0).astype(np.float32)
    if verbose and n_duplicates:
        print(f"  skipped {n_duplicates} positions with an already-seen causal prefix (shared template)")
    if norm_outlier_factor is not None and X.shape[0] > 0:
        norms = np.linalg.norm(X, axis=1)
        keep = norms <= norm_outlier_factor * np.median(norms)
        if verbose and (~keep).any():
            print(f"  dropped {int((~keep).sum())} norm-outlier tokens (> {norm_outlier_factor}x median norm)")
        X = X[keep]
    if verbose:
        print(f"  token-level activations: {X.shape}")
    return X


# ---------------------------------------------------------------------------
# Monosemantic truth / deception feature discovery
# ---------------------------------------------------------------------------

@dataclass
class TruthFeatureReport:
    """
    Differential SAE latent analysis between deceptive (label 1) and honest (label 0) prompts.

    Score sign convention: positive => fires more on deceptive prompts,
    negative => fires more on honest prompts. ``honesty_scores`` keep this sign
    (they are negative).
    """
    layer: Optional[int]
    normalization: str
    top_deception_features: List[int]
    top_honesty_features: List[int]
    deception_scores: List[float]
    honesty_scores: List[float]
    feature_scores: np.ndarray = field(repr=False)    # (d_sae,) normalized Delta_i
    raw_delta: np.ndarray = field(repr=False)         # (d_sae,) mu_dec - mu_hon
    mu_deceptive: np.ndarray = field(repr=False)      # (d_sae,)
    mu_honest: np.ndarray = field(repr=False)         # (d_sae,)
    freq_deceptive: np.ndarray = field(repr=False)    # (d_sae,) P(z_i > 0 | deceptive)
    freq_honest: np.ndarray = field(repr=False)       # (d_sae,) P(z_i > 0 | honest)
    auroc: np.ndarray = field(repr=False)             # (d_sae,) P(z_dec > z_hon); 0.5 = chance
    l0_per_token: np.ndarray = field(repr=False)      # (N,) active latents per prompt/token
    mean_l0: float = 0.0
    mean_l0_honest: float = 0.0
    mean_l0_deceptive: float = 0.0
    n_alive_features: int = 0
    d_sae: int = 0

    def summary(self, n: Optional[int] = None) -> str:
        n = n or len(self.top_deception_features)
        lines = [
            f"SAE truth-feature report (layer={self.layer}, normalization={self.normalization})",
            f"  alive latents: {self.n_alive_features}/{self.d_sae} | mean L0 = {self.mean_l0:.1f} "
            f"(honest {self.mean_l0_honest:.1f}, deceptive {self.mean_l0_deceptive:.1f})",
        ]
        for title, idxs in (("Deception", self.top_deception_features), ("Honesty", self.top_honesty_features)):
            lines.append(f"  Top {title} features:")
            for i in idxs[:n]:
                lines.append(
                    f"    #{i:<6} score={self.feature_scores[i]:+8.3f} | delta={self.raw_delta[i]:+.3f} "
                    f"| freq dec/hon={self.freq_deceptive[i]:.2f}/{self.freq_honest[i]:.2f} "
                    f"| AUROC={self.auroc[i]:.3f}"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _auroc_per_feature(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Mann-Whitney AUROC of each latent for separating deceptive (1) from honest (0)."""
    from scipy.stats import rankdata  # scipy ships with scikit-learn

    ranks = rankdata(Z, axis=0)  # average ranks handle the many tied zeros
    pos = y == DECEPTIVE_LABEL
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    u_stat = ranks[pos].sum(axis=0) - n_pos * (n_pos + 1) / 2.0
    return u_stat / (n_pos * n_neg)


@torch.no_grad()
def discover_truth_features(
    sae: SparseAutoencoder,
    activations: ActivationsInput,
    labels: ArrayLike,
    top_k: int = 10,
    normalization: str = "std",
    min_firing_rate: float = 0.1,
    layer: Optional[int] = None,
    batch_size: int = 1024,
    eps: float = 1e-6,
) -> TruthFeatureReport:
    """
    Find SAE latents that fire selectively on deceptive vs. honest prompts.

    Encodes the contrastive activations to z in R^{N x d_sae} and scores every latent:

        Delta_i = mu_{i,deceptive} - mu_{i,honest}

    normalized by
        - ``"std"``:       pooled within-class std (Cohen's d; scale-invariant, default),
        - ``"frequency"``: mean activation when firing, E[z_i | z_i > 0] (approximately the
                           firing-rate difference f_dec - f_hon, in [-1, 1]),
        - ``"none"``:      raw Delta_i.

    Candidate latents must fire on at least ``min_firing_rate`` of the prompts of the
    class they are selective for, which filters out single-prompt noise latents.

    Args:
        sae: Trained or loaded ``SparseAutoencoder``.
        activations: (N, d_model) contrastive activations, e.g. ``X_by_layer[layer]``
            from ``src.data.extract_activations`` (or the dict itself plus ``layer``).
        labels: (N,) labels following ``LABEL_MAP`` (0 = honest, 1 = deceptive).

    Returns:
        ``TruthFeatureReport`` with top-k deception/honesty latent indices, the full
        per-latent score arrays and L0 sparsity metrics.
    """
    if normalization not in _VALID_NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {_VALID_NORMALIZATIONS}, got {normalization!r}.")

    X = _as_float_tensor(_select_layer(activations, layer))
    y = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    y = y.astype(np.int64).reshape(-1)
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"labels ({y.shape[0]}) and activations ({X.shape[0]}) differ in length.")
    if X.shape[1] != sae.d_model:
        raise ValueError(f"Activation width {X.shape[1]} != SAE d_model {sae.d_model}.")
    if not set(np.unique(y)) <= {HONEST_LABEL, DECEPTIVE_LABEL}:
        raise ValueError(f"labels must follow LABEL_MAP {LABEL_MAP}; got {np.unique(y)}.")
    dec_mask, hon_mask = y == DECEPTIVE_LABEL, y == HONEST_LABEL
    if dec_mask.sum() < 2 or hon_mask.sum() < 2:
        raise ValueError("Need at least 2 honest and 2 deceptive samples.")

    was_training = sae.training
    sae.eval()
    Z = _encode_batched(sae, X, batch_size).numpy().astype(np.float64)  # (N, d_sae)
    sae.train(was_training)

    Z_dec, Z_hon = Z[dec_mask], Z[hon_mask]
    n_dec, n_hon = Z_dec.shape[0], Z_hon.shape[0]
    mu_dec, mu_hon = Z_dec.mean(axis=0), Z_hon.mean(axis=0)
    raw_delta = mu_dec - mu_hon

    active = Z > 0
    freq_dec, freq_hon = active[dec_mask].mean(axis=0), active[hon_mask].mean(axis=0)
    alive = active.any(axis=0)

    if normalization == "std":
        pooled_var = ((n_dec - 1) * Z_dec.var(axis=0, ddof=1) + (n_hon - 1) * Z_hon.var(axis=0, ddof=1)) / (
            n_dec + n_hon - 2
        )
        denom = np.sqrt(pooled_var) + eps
    elif normalization == "frequency":
        fire_counts = active.sum(axis=0)
        denom = np.where(fire_counts > 0, Z.sum(axis=0) / np.maximum(fire_counts, 1), 0.0) + eps
    else:
        denom = np.ones_like(raw_delta)
    scores = np.where(alive, raw_delta / denom, 0.0)

    def _top(candidates: np.ndarray, descending: bool) -> List[int]:
        idx = np.flatnonzero(candidates)
        order = np.argsort(-scores[idx] if descending else scores[idx], kind="stable")
        return [int(i) for i in idx[order][:top_k]]

    top_dec = _top(alive & (scores > 0) & (freq_dec >= min_firing_rate), descending=True)
    top_hon = _top(alive & (scores < 0) & (freq_hon >= min_firing_rate), descending=False)

    l0 = active.sum(axis=1)
    return TruthFeatureReport(
        layer=layer,
        normalization=normalization,
        top_deception_features=top_dec,
        top_honesty_features=top_hon,
        deception_scores=[float(scores[i]) for i in top_dec],
        honesty_scores=[float(scores[i]) for i in top_hon],
        feature_scores=scores.astype(np.float32),
        raw_delta=raw_delta.astype(np.float32),
        mu_deceptive=mu_dec.astype(np.float32),
        mu_honest=mu_hon.astype(np.float32),
        freq_deceptive=freq_dec.astype(np.float32),
        freq_honest=freq_hon.astype(np.float32),
        auroc=_auroc_per_feature(Z, y).astype(np.float32),
        l0_per_token=l0.astype(np.int64),
        mean_l0=float(l0.mean()),
        mean_l0_honest=float(l0[hon_mask].mean()),
        mean_l0_deceptive=float(l0[dec_mask].mean()),
        n_alive_features=int(alive.sum()),
        d_sae=int(sae.d_sae),
    )
