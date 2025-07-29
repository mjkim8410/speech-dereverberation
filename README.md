# 🗣️ Audio Dereverberation with Conv‑TasNet

> End‑to‑end, time‑domain speech dereverberation at 16 kHz using a Conv‑TasNet‑style separator with dilated TCN blocks and SI‑SDR training.

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![torchaudio](https://img.shields.io/badge/torchaudio-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Loss & Metrics](#loss--metrics)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Export & Inference](#export--inference)
- [Tips for Stability & Speed](#tips-for-stability--speed)
- [Results Placeholder](#results-placeholder)
- [Roadmap](#roadmap)
- [Citations](#citations)
- [License](#license)
- [Ethics & Intended Use](#ethics--intended-use)

---

## Overview

This repo trains a **single‑channel speech dereverberation** model directly in the **time domain**. It follows the Conv‑TasNet idea: learn an analysis filterbank, enhance with stacked dilated temporal convolution (TCN) blocks, then resynthesize via a synthesis filterbank. Conv‑TasNet has proven highly effective for time‑domain source separation, and the same mechanics transfer to dereverberation.

**Key features**
- **Time‑domain** end‑to‑end training (no STFT required)
- **Dilated TCN** separator with residual connections
- **Negative SI‑SDR** training objective
- Mixed precision (AMP) and optional **8‑bit Adam/AdamW** (bitsandbytes) to cut optimizer memory
- Simple dataset interface for paired *(reverb, clean)* audio segments

---

## Model Architecture

**High level**
1. **Encoder (Conv1d):** learned analysis filterbank  
2. **Separator (TCN stacks):** repeated blocks  
   - 1×1 bottleneck → **depth‑wise dilated Conv1d** (increasing dilation per layer) → **PReLU** → **Global Channel LayerNorm** → 1×1 point‑wise  
   - residual connection and mask estimation
3. **Masking:** element‑wise masks (sigmoid) on encoded mixture
4. **Decoder (ConvTranspose1d):** overlap‑add synthesis back to waveform

**Default hyper‑parameters (baseline)**
```python
num_sources=1
encoder_kernel_size=16      # L
encoder_stride=8            # 50% overlap (L/2)
encoder_filters=512         # N
tcn_hidden=128              # B
tcn_kernel_size=3
tcn_layers=8                # per stack (dilations 1..128)
tcn_stacks=3                # number of repeats
causal=False

















├─ src/
│  ├─ model.py            # ConvTasNet + TCN blocks + GlobalChannelLayerNorm
│  ├─ train.py            # training loop
│  ├─ dereverb.py         # inference / export helpers
│  └─ utils.py            # I/O and helpers (optional)
├─ data/
│  ├─ reverb_chunks/      # reverberant inputs (10 s segments)
│  └─ clean_chunks/       # clean targets (matched names)
├─ checkpoints/           # saved model weights (.pth)
├─ README.md
└─ LICENSE













