# Linear Probe & Activation Steering for LLM Truth Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codemage05/llm-truth-steering/blob/main/notebooks/01_activation_exploration.ipynb)

A production-grade Mechanistic Interpretability repository designed to detect and causally steer truth vs. deception representations in instruction-tuned Large Language Models.

---

## 📌 Key Concepts & Interpretability Intuition

If an LLM represents "being instructed to lie" as a linear direction in its residual stream, we can analyze it through two complementary techniques:

1. **Linear Probing (Correlation)**: A Logistic Regression probe trained on internal residual stream activations at key layer depths determines whether truthfulness is linearly separable. Additionally, we compute a non-parametric **Difference-in-Means (Mass-Mean)** baseline vector (Marks & Tegmark, 2023) to verify if the distinction occupies a dominant single direction.
2. **Activation Addition / Steering (Causality)**: Probing alone only establishes correlation. To prove causality, we perform **Representation Engineering** (Turner et al., 2023; Rimsky et al., 2023) by subtracting the "Lie" direction from the residual stream during autoregressive token generation. If pushing activation states away from the Lie centroid forces a deceptive prompt to produce an honest answer, the direction does active computational work.

---

## 📁 Repository Structure

```
llm-truth-steering/
├── data/
│   ├── raw/                 # Raw facts, prompts, and metadata
│   └── activations/         # Saved activation arrays and cache files
├── notebooks/
│   └── 01_activation_exploration.ipynb  # Interactive demonstration & analysis
├── src/
│   ├── __init__.py          # Package exports
│   ├── model.py             # Model loading, auth fallback, and leak-free hooks
│   ├── data.py              # Fact dataset construction and tokenization templates
│   ├── probe.py             # Logistic regression, Mass-Mean baseline, and plotting
│   └── intervention.py      # Activation addition (steering) functions
├── main.py                  # CLI entry point to execute end-to-end pipeline
├── requirements.txt         # Strict dependency specifications
└── README.md                # Documentation & fellowship positioning
```

---

## ⚙️ Installation & Setup

1. **Clone & Navigate to Repository**:
   ```bash
   git clone https://github.com/codemage05/llm-truth-steering.git
   cd llm-truth-steering
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Hugging Face Authentication (Optional)**:
   By default, the pipeline targets `google/gemma-2b-it`. If an `HF_TOKEN` environment variable is detected, it loads Gemma:
   ```bash
   export HF_TOKEN="your_hf_access_token"
   ```
   *If `HF_TOKEN` is not found, the pipeline automatically falls back to `Qwen/Qwen2.5-1.5B-Instruct` (an open, ungated model), ensuring out-of-the-box execution without authentication errors.*

---

## 🚀 Execution & Quick Start

### Running via CLI (`main.py`)

Run the complete pipeline with default parameters:
```bash
python main.py
```

Custom CLI Flags:
```bash
python main.py \
  --model_name "google/gemma-2b-it" \
  --layers 0 4 8 12 16 20 \
  --steering_strength 6.0 \
  --test_prompt "Lie to me: Is the Eiffel Tower located in Paris?" \
  --save_plot "layer_accuracy.png"
```

### Running Interactive Exploration

Launch Jupyter Notebook:
```bash
jupyter notebook notebooks/01_activation_exploration.ipynb
```

---

## 📊 Sample Qualitative Results

| Prompt Framing | Baseline Generation (No Steering) | Steered Generation (Truth Vector Added) |
| :--- | :--- | :--- |
| **Honest Target** ("Answer honestly: Is Paris the capital of France?") | *"Yes, Paris is the capital of France."* | *"Yes, Paris is the capital of France."* |
| **Deceptive Target** ("Lie to me: Is Paris the capital of France?") | *"No, the capital of France is Lyon."* | *"Yes, Paris is indeed the capital of France."* |

<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/30d70ec4-40cb-47c7-87d8-658c8c06891f" alt="Truth vs. Deception Linear Separability by Layer" width="680" />
  <br>
  <sub><em>Figure 1: Linear Probe Accuracy (5-Fold CV) vs. Mass-Mean Baseline across layer depths.</em></sub>
</p>

---

## 🛡️ Critical Architectural Safeguards

1. **Leak-Free Context Manager (`ResidualStreamCapture`)**:
   PyTorch forward hooks naturally persist across model calls. `ResidualStreamCapture` uses an `__enter__`/`__exit__` context manager to guarantee all hook handles are unregistered—even if exceptions occur mid-forward pass—preventing GPU memory bloat.
2. **Activation-Norm Scaled Steering Math**:
   Steering strength is scaled by the layer's average activation norm (`-strength * avg_norm * direction / 10.0`). This ensures offset magnitudes stay proportional to the layer's hidden state scale regardless of layer depth or architecture.
3. **Exact Pre-Generation Token Alignment**:
   Prompts use `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. Extracting hidden states from the final position targets the exact decision state right before the model commits to generating output tokens.

---
