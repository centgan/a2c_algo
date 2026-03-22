# A2C Multi-Agent Trading Bot

An Advantage Actor-Critic (A2C) reinforcement learning agent for NAS100 trading, with support for **single-GPU** and **dual-GPU (2× RTX 3090)** training.

## Architecture

The system uses a **multi-process architecture**:

- **Agent Workers** (default 32) — each runs an environment simulation and generates experiences, using a lightweight copy of the model for inference.
- **Learner** — a central process that consumes batched experiences from shared memory and trains the global model. When using 2 GPUs, the learner distributes training across both via TensorFlow's `MirroredStrategy`.
- **Weight Sync** — the learner periodically saves updated weights to disk (`.npy`), which agents poll and load to stay in sync.

```
┌──────────────┐  ┌──────────────┐        ┌──────────────┐
│  Agent 0     │  │  Agent 1     │  ...   │  Agent N     │
│  (env + inf) │  │  (env + inf) │        │  (env + inf) │
└──────┬───────┘  └──────┬───────┘        └──────┬───────┘
       │                 │                       │
       └────────┬────────┴───────────────────────┘
                │   shared memory (experiences)
         ┌──────▼──────┐
         │   Learner   │
         │  (1 or 2    │
         │   GPUs)     │
         └─────────────┘
```

## Requirements

```
tensorflow>=2.13.0
tensorflow-probability>=0.21.0
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=12.0.0
loguru>=0.7.0
tqdm>=4.65.0
```

Install:
```bash
pip install -r requirements.txt
```

> **RTX Note**: Make sure you have the NVIDIA driver ≥ 525 and CUDA 11.8+ / 12.x installed. TensorFlow 2.13+ supports RTX 30-series and 40-series out of the box.

## Usage

```bash
python main.py [OPTIONS]
```

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--num-gpus` | `1` | Number of GPUs: `1` (single) or `2` (dual with MirroredStrategy) |
| `--num-agents` | `32` | Number of parallel agent worker processes |
| `--batch-size` | `256` | Batch size for learner updates |
| `--epochs` | `2` | Number of training epochs |
| `--resume` | `True` | Resume from last checkpoint |
| `--no-resume` | — | Start fresh training, ignore existing checkpoints |
| `--gpu-memory-limit` | `None` | Per-GPU memory cap in MB (e.g. `20000` for 24 GB RTX 3090) |
| `--mixed-precision` | `False` | Enable FP16 mixed precision for faster training on RTX tensor cores |

### Examples

**Single GPU (default)**
```bash
python main.py
```

**Single GPU with memory limit (useful when sharing the GPU)**
```bash
python main.py --gpu-memory-limit 20000
```

**Dual RTX 3090 training**
```bash
python main.py --num-gpus 2
```

**Dual GPU with mixed precision and fresh start**
```bash
python main.py --num-gpus 2 --mixed-precision --no-resume
```

**Fewer agents for testing**
```bash
python main.py --num-agents 4 --num-gpus 1
```

**Show all options**
```bash
python main.py --help
```

## GPU Details

### Memory Management

By default, TensorFlow uses **dynamic memory growth** — it only allocates GPU memory as needed. This prevents the common RTX issue where TF pre-allocates all 24 GB of VRAM upfront.

If you need tighter control (e.g. running other processes on the same GPU), use `--gpu-memory-limit`:

```bash
# Leave ~4 GB free on each RTX 3090
python main.py --gpu-memory-limit 20000
```

### Multi-GPU (MirroredStrategy)

When `--num-gpus 2` is set:

1. The **learner process** creates the Actor and Critic networks inside a `tf.distribute.MirroredStrategy` scope.
2. Each training batch is automatically split across both GPUs.
3. Gradients are synchronized via NCCL (all-reduce) after each step.
4. Agent workers still use a single GPU for inference — they're lightweight and don't need distribution.

If 2 GPUs are requested but only 1 is detected, the system automatically falls back to single-GPU mode with a warning.

### Mixed Precision

RTX 30-series (Ampere) and 40-series (Ada Lovelace) GPUs have dedicated FP16 tensor cores. Enable `--mixed-precision` to use them:

```bash
python main.py --mixed-precision
```

This uses `float16` for compute and `float32` for accumulation, typically giving **1.5–2× speedup** with minimal impact on training quality.

## Checkpointing

Training state is automatically saved:

- **Every 1000 learner updates** — model weights + best model
- **Every 10 minutes** — full learner state (update count, reward history, best reward)
- **On Ctrl+C** — graceful shutdown with checkpoint save

Resume with `--resume` (default) or start fresh with `--no-resume`.

Checkpoints are saved to `./checkpoints/`.

## Project Structure

```
a2c_algo/
├── main.py                  # Entry point — CLI, GPU config, multiprocessing
├── src/
│   ├── model.py             # Agent class (Actor-Critic with optional multi-GPU)
│   ├── network.py           # LSTM Actor & Critic network definitions
│   ├── environment.py       # Trading environment with batch processing
│   ├── checkpoint_manager.py # Checkpoint save/load utilities
│   └── indicators.py        # Technical indicators (RSI, MACD, OB, FVG, news)
├── visualize_results.py     # Plot training results
├── setup_scheduler.py       # Scheduled training launcher
├── requirements.txt
└── README.md
```
