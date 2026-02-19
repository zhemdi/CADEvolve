# CADEvolve: Creating Realistic CAD via Program Evolution

**CADEvolve** is an evolution-based pipeline, dataset, and training codebase for generating **industrial-grade, valid CAD programs** and fine-tuning **vision–language models (VLMs)** for **Image2CAD**.

**Paper:** https://arxiv.org/abs/2602.16317  
**Dataset:** https://huggingface.co/datasets/kulibinai/cadevolve  
**Model:** https://huggingface.co/kulibinai/cadevolve-rl1




---

## Repository layout

```text
CADEvolve/
├── dataset_utils/  # post-processing + augmentation
├── evolution/      # evolution pipeline to grow ~8k complex CadQuery generators
└── train/          # SFT training, inference, and evaluation for Image2CAD VLMs
```

### High-level flow

1. **Evolution** (`evolution/`) starts from a small pool of hand-written CadQuery generators and incrementally grows program complexity using **VLM-guided edits** plus **geometry validation**, producing **≈8k complex parts** (executable CadQuery generators).
2. **Dataset expansion** (`dataset_utils/`) applies multi-stage post-processing and augmentation to expand those generators into a unified dataset of **≈1.3M scripts** paired with multi-view renders.
3. **Model training & evaluation** (`train/`) fine-tunes a VLM (SFT), runs inference, and computes Image2CAD metrics on standard benchmarks.


## Quick start

> This README gives the top-level entry points. Each folder contains self-contained scripts.  
> `dataset_utils/` includes its own README with the exact post-processing/augmentation pipeline.


### 0) Installation

```bash

pip install -r requirements.txt
```


### 1) Evolve complex generators (≈8k parts)

Go to `evolution/` and run the evolution pipeline.

**Key files**
- `cadquery_examples.txt` — initial pool of **46 hand-written** CadQuery scripts (seeds).
- `geometry_check.py` — geometry validity checks used during evolution.
- `pipeline.py` — main script that runs the evolution procedure (VLM-guided edits + validations).

**Typical usage**
```bash
cd evolution
python pipeline.py
```

**Outputs**: a set of validated CadQuery parametric generators representing evolved parts.

### 2) Expand to the full dataset

`dataset_utils/` takes the evolved generators and applies multi-stage post-processing and augmentation to create the final dataset.

```bash
cd dataset_utils
# See dataset_utils/README.md for the exact pipeline and arguments
``` 

### 3) Train / infer / evaluate (Image2CAD VLM)

All training utilities live in `train/`.

**Key files:**

- `train.py` — supervised fine-tuning (SFT).
- `inference.py` — inference script for generating CadQuery code from images.
- `eval.py` — metric computation for Image2CAD evaluation.
- `config.yaml` — training configuration.
- `visualization/`, `visualization_norm.py` — multi-view rendering utilities used to generate model inputs (multi-view images).

## What you get

- **Evolution-generated CAD**: ≈8k complex, valid parts as executable CadQuery parametric generators.
- **Unified large-scale dataset**: ≈1.3M CadQuery scripts paired with multi-view renders, covering the full CadQuery operation set.
- **Training code**: SFT + inference + evaluation for Image2CAD VLMs, including rendering utilities for consistent multi-view inputs.


## Links

- Dataset: [CADEvolve Hugging Face dataset](https://huggingface.co/datasets/kulibinai/cadevolve)
- Model: [CADEvolve Hugging Face model](https://huggingface.co/kulibinai/cadevolve-rl1)


Citation
```bibtex
@article{elistratov2026cadevolve,
  title={CADEvolve: Creating Realistic CAD via Program Evolution},
  author={Elistratov, Maksim and Barannikov, Marina and Ivanov, Gregory and Khrulkov, Valentin and Konushin, Anton and Kuznetsov, Andrey and Zhemchuzhnikov, Dmitrii},
  journal={arXiv preprint arXiv:2602.16317},
  year={2026}
}
```
