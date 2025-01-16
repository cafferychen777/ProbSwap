# ProbSwap: A Probability-Based Attack on LLM Watermarks

ProbSwap is an attack method targeting LLM watermarks by identifying and replacing low-probability tokens that are likely introduced by watermarking mechanisms.

## Core Idea

LLM watermarking typically works by modifying the sampling probability distribution during text generation to embed statistical patterns. ProbSwap attacks these watermarks by:

1. Identifying tokens that have relatively low generation probability
2. Replacing these tokens with semantically similar alternatives that have higher probability or appear more natural to another LLM

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

- `probswap/`
  - `attack.py`: Core implementation of the ProbSwap attack
  - `models.py`: Interface with different LLM models
  - `claude_wrapper.py`: Claude API integration for substitute generation
  - `markllm_integration.py`: Integration with MarkLLM watermarking toolkit
  - `utils.py`: Utility functions

- `experiments/`
  - `run_watermark_experiments.py`: Run watermarking attack experiments
  - `run_experiments.py`: Run general attack experiments

- `evaluation/`
  - `evaluate_watermark.py`: Evaluate attack effectiveness on watermarks
  - `evaluate.py`: General evaluation metrics

## Usage

### Using Local Models

```python
from probswap.attack import ProbSwapAttack
from probswap.models import ModelWrapper

# Initialize models
target_model = ModelWrapper("your-watermarked-model-name")
substitute_model = ModelWrapper("your-substitute-model-name")

# Initialize attack
attack = ProbSwapAttack(
    target_model=target_model.model,
    target_tokenizer=target_model.tokenizer,
    substitute_model=substitute_model,  # Local model wrapper
    prob_threshold=0.1,
    top_k_substitutes=5
)

# Apply attack
modified_text, modifications = await attack.attack(watermarked_text)
```

### Using Claude API

```python
from probswap.attack import ProbSwapAttack
from probswap.claude_wrapper import ClaudeWrapper

# Initialize models
target_model = ModelWrapper("your-watermarked-model-name")
substitute_model = ClaudeWrapper(target_tokenizer=target_model.tokenizer)

# Initialize attack
attack = ProbSwapAttack(
    target_model=target_model.model,
    target_tokenizer=target_model.tokenizer,
    substitute_model=substitute_model,  # Claude API wrapper
    prob_threshold=0.1,
    top_k_substitutes=5
)

# Apply attack
modified_text, modifications = await attack.attack(watermarked_text)
```

## Environment Variables

When using the Claude API, you need to set your API key in a `.env` file:

```bash
ANTHROPIC_API_KEY=your-api-key
```

## Evaluation

Run watermarking experiments:
```bash
python experiments/run_watermark_experiments.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
