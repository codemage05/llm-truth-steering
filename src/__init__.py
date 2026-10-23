"""
Linear Probe Portfolio package for Mechanistic Interpretability.
"""

from .model import load_model_and_tokenizer, ResidualStreamCapture
from .data import (
    FACTS,
    EXTRA_FACTS,
    ALL_FACTS,
    INSTRUCTIONS,
    INSTRUCTION_TEMPLATES,
    HELDOUT_TEMPLATE_IDS,
    LABEL_MAP,
    format_chat,
    build_dataset,
    extract_activations,
)
from .probe import LayerResult, mass_mean_direction, mass_mean_accuracy, train_probes, plot_layer_accuracy
from .intervention import steer_generation
from .sae import (
    SparseAutoencoder,
    SAEMetrics,
    TruthFeatureReport,
    train_sae,
    evaluate_sae,
    load_or_train_sae,
    extract_token_activations,
    discover_truth_features,
)
from .sae_intervention import (
    ResidualStreamEdit,
    DenseSteeringEdit,
    SAEFeatureClampHook,
    SparseSteeringVector,
    sae_feature_clamp_hook,
    compute_sae_steering_vector,
    make_sparse_layer_result,
    residual_edit_hook,
    steer_generation_with_edit,
    generate_batch,
)

__all__ = [
    "load_model_and_tokenizer",
    "ResidualStreamCapture",
    "FACTS",
    "EXTRA_FACTS",
    "ALL_FACTS",
    "INSTRUCTIONS",
    "INSTRUCTION_TEMPLATES",
    "HELDOUT_TEMPLATE_IDS",
    "LABEL_MAP",
    "format_chat",
    "build_dataset",
    "extract_activations",
    "LayerResult",
    "mass_mean_direction",
    "mass_mean_accuracy",
    "train_probes",
    "plot_layer_accuracy",
    "steer_generation",
    "SparseAutoencoder",
    "SAEMetrics",
    "TruthFeatureReport",
    "train_sae",
    "evaluate_sae",
    "load_or_train_sae",
    "extract_token_activations",
    "discover_truth_features",
    "ResidualStreamEdit",
    "DenseSteeringEdit",
    "SAEFeatureClampHook",
    "SparseSteeringVector",
    "sae_feature_clamp_hook",
    "compute_sae_steering_vector",
    "make_sparse_layer_result",
    "residual_edit_hook",
    "steer_generation_with_edit",
    "generate_batch",
]

