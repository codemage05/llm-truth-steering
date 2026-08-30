"""
Linear Probe Portfolio package for Mechanistic Interpretability.
"""

from .model import load_model_and_tokenizer, ResidualStreamCapture
from .data import FACTS, INSTRUCTIONS, LABEL_MAP, build_dataset, extract_activations
from .probe import LayerResult, mass_mean_direction, mass_mean_accuracy, train_probes, plot_layer_accuracy
from .intervention import steer_generation

__all__ = [
    "load_model_and_tokenizer",
    "ResidualStreamCapture",
    "FACTS",
    "INSTRUCTIONS",
    "LABEL_MAP",
    "build_dataset",
    "extract_activations",
    "LayerResult",
    "mass_mean_direction",
    "mass_mean_accuracy",
    "train_probes",
    "plot_layer_accuracy",
    "steer_generation",
]
