# Assignment 2 – Mini GPT Training Pipeline

This folder contains a compact, fully reproducible training pipeline for a GPT‑style language model trained on the tokenised dataset built in Assignment 1.

## Project Layout
- `mini_gpt.py` – decoder-only transformer model (`MiniGPT`) and configuration dataclass.
- `data_utils.py` – dataset loader that converts the saved `.pt` batches from Assignment 1 into individual token sequences and supplies a padding collate function.
- `train_mini_gpt.py` – end-to-end training script (argument parsing, training loop, evaluation, checkpointing, logging).
- `plot_metrics.py` – helper for visualising loss and perplexity.

## Prerequisites
- Python ≥ 3.10
- `torch` (CPU build is sufficient), `matplotlib`, `tqdm` (optional but recommended)

Install them via:
```bash
pip install torch matplotlib tqdm
```

## Running a Training Experiment
```bash
python assignment2/train_mini_gpt.py \
  --dataset assignment1/sample_dataset.pt \
  --output-dir assignment2/runs/exp01 \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 5e-4 \
  --embed-dim 128 \
  --num-layers 2 \
  --num-heads 4 \
  --dropout 0.1 \
  --save-every 1000
```

Outputs written to `--output-dir`:
- `mini_gpt_epochXXX.pt` – rolling checkpoints (model + optimiser state + history + config)
- `mini_gpt_best.pt` – latest checkpoint with the lowest validation loss so far
- `mini_gpt_last.pt` – most recent checkpoint (updated even on Ctrl+C interruptions)
- `training_log.json` – per-epoch loss/perplexity

The script also supports:
- `--max-seq-len` to truncate long token blocks for efficiency
- `--val-split` to adjust validation proportion
- `--save-every` to dump intermediate checkpoints during an epoch
- `--grad-clip` to avoid exploding gradients
- `--num-workers` to parallelise DataLoader preprocessing (defaults to half your logical CPU cores)

Notes:
- When running on CUDA, training automatically enables TF32, mixed precision (AMP), and flash/efficient attention kernels when available.

## Visualising Training Curves
After training, generate plots:
```bash
python assignment2/plot_metrics.py \
  --log-file assignment2/runs/exp01/training_log.json \
  --output assignment2/runs/exp01/metrics.png
```

## Hyperparameter Experiments
Suggested sweeps:
1. **Learning rate**: `{1e-3, 5e-4, 1e-4}`
2. **Batch size**: `{8, 16, 32}` (adjust gradient clipping if batches are small)
3. **Model depth**: `{1, 2}` layers and `{2, 4}` attention heads (ensure `embed_dim % num_heads == 0`)
4. **Embedding size**: `{64, 128, 256}` – monitor GPU/CPU memory usage

Record each run’s configuration together with the corresponding `training_log.json` excerpts and plots in your report.

## Checkpoint Usage
Resume training from a saved checkpoint:
```python
import torch
from assignment2.mini_gpt import MiniGPT, MiniGPTConfig

payload = torch.load("assignment2/runs/exp01/mini_gpt_epoch003.pt", map_location="cpu")
config = MiniGPTConfig(**payload["config"])
model = MiniGPT(config)
model.load_state_dict(payload["model_state_dict"])
model.eval()
```

Use `model.generate(...)` for qualitative inspection of samples (after mapping token IDs back to text with the tokenizer from Assignment 1).
