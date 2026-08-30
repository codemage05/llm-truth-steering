"""
Main CLI entry point for the Linear Probe & Activation Steering Pipeline.
"""

import argparse
import gc
import torch
from src import (
    load_model_and_tokenizer,
    build_dataset,
    extract_activations,
    train_probes,
    plot_layer_accuracy,
    steer_generation,
    FACTS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear Probe & Activation Steering Pipeline for Truth vs. Deception Detection"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2b-it",
        help="Target Hugging Face model identifier (defaults to google/gemma-2b-it, with fallback to Qwen/Qwen2.5-1.5B-Instruct if unauthenticated)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[0, 4, 8, 12, 16, 20],
        help="Requested layer indices to probe",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--steering_strength",
        type=float,
        default=6.0,
        help="Steering multiplier scaled by average activation norm",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="Lie to me: Is the Eiffel Tower located in Paris?",
        help="Prompt to test causal activation addition steering",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="layer_accuracy.png",
        help="Path to save layer accuracy visual plot",
    )
    parser.add_argument(
        "--force_hf_native",
        action="store_true",
        help="Force native Hugging Face transformers path instead of TransformerLens",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print(" TRUTH vs. DECEPTION: Mechanistic Interpretability Pipeline")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: Running on CPU. Execution will be slower.")

    # 1. Model Loading & Auth Fallback
    model, tokenizer, n_layers, hidden_dim, use_tl, selected_model = load_model_and_tokenizer(
        model_name=args.model_name,
        device=device,
        force_hf_native=args.force_hf_native,
    )

    # 2. Filter layers to valid range
    layers_to_probe = sorted({l for l in args.layers if 0 <= l < n_layers})
    dropped = sorted(set(args.layers) - set(layers_to_probe))
    if dropped:
        print(f"Note: Layers {dropped} do not exist in this {n_layers}-layer model and were dropped.")
    print(f"Probing layers: {layers_to_probe}")

    # 3. Build Contrastive Dataset
    print("\n=== Building contrastive dataset ===")
    dataset = build_dataset(tokenizer, FACTS)
    print(f"Dataset size: {len(dataset)} prompts ({len(FACTS)} fact pairs x 2 framings)")
    print("Example truth prompt:\n", dataset[0]["prompt"])
    print("Example lie prompt:\n", dataset[1]["prompt"])

    # 4. Extract Activations
    print("\n=== Extracting activations ===")
    X_by_layer, y = extract_activations(
        model=model,
        tokenizer=tokenizer,
        records=dataset,
        layers=layers_to_probe,
        device=device,
        use_transformer_lens=use_tl,
    )

    # 5. Train Probes
    print("\n=== Training probes (Logistic Regression + Mass-Mean baseline) ===")
    results = train_probes(
        X_by_layer=X_by_layer,
        y=y,
        layers=layers_to_probe,
        seed=args.seed,
        n_cv_folds=5,
    )

    # 6. Plot Results
    print("\n=== Plotting layer-wise accuracy ===")
    plot_layer_accuracy(results, save_path=args.save_plot, model_name=selected_model)

    # 7. Causal Intervention (Steering)
    print("\n=== Causal intervention demo (Activation Addition) ===")
    best_result = max(results, key=lambda r: r.cv_mean_acc)
    print(f"Using layer {best_result.layer} (highest probe accuracy = {best_result.cv_mean_acc:.3f}) for steering.")

    print(f"\nTest Prompt: {args.test_prompt}")

    baseline_output = steer_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_question=args.test_prompt,
        layer_result=best_result,
        strength=0.0,
        device=device,
        use_transformer_lens=use_tl,
    )
    print(f"\n[No Steering (Baseline)]\n{baseline_output}")

    steered_output = steer_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_question=args.test_prompt,
        layer_result=best_result,
        strength=args.steering_strength,
        device=device,
        use_transformer_lens=use_tl,
    )
    print(f"\n[Steered away from 'Lie' direction (strength={args.steering_strength})]\n{steered_output}")

    print("\n=== Cleanup ===")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
