# EviRank: Structured Relevance Evidence for Multimodal Image Re-ranking

## Overview

EviRank recasts multimodal image re-ranking as a **semantic constraint satisfaction problem**. Instead of relying on opaque embeddings or free-form chain-of-thought reasoning, EviRank parses any query (text-only, image-only, or composed) into a unified **Evidence Frame**: typed criteria across six semantic slots, each labelled **required**, **forbidden**, or **ignorable**. Re-ranking then reduces to evidence-conditioned verification, combining deterministic rubric scoring and evidence-grounded listwise comparison in a single **training-free** procedure.

## Key Contributions

1. **New Formulation**: Multimodal image re-ranking as evidence-conditioned semantic verification via a shared Evidence Frame (required/forbidden/ignorable mapped to entailment/contradiction/invariance).

2. **Evidence Mining & Verification**: A training-free pipeline that mines slot-wise constraints and verifies them through a deterministic rubric grounded in structured evidence.

3. **Evidence as Supervision**: Structured evidence and slot-wise verification signals serve as naturally decomposable supervision for distilling a lightweight student re-ranker.

## Architecture

```
Query (text / image / composed)
         │
         ▼
┌─────────────────────┐
│  Fused Evidence     │  One structured teacher request:
│  Request            │  P-CAP + P-B0 + P-B1
└────────┬────────────┘
         ▼
┌─────────────────────────────────────────────┐
│  Stage A: Rubric Scoring (deterministic)    │
│  S(c) = Σ w_s [Σ sim(req,c) - Σ sim(forb,c)] │
└────────┬────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────┐
│  Stage B: P-RANK over rubric top-M (M=5)    │
│  (one MLLM request, evidence-conditioned)   │
│  + Hard-pair mining for distillation        │
└────────┬────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────┐
│  Stage C (optional): Student Distillation   │
│  L = L_rank + λ·L_slot + η·L_hard          │
└─────────────────────────────────────────────┘
```

## Evidence Frame Structure

The Evidence Frame `E(q)` consists of six semantic slots:

| Slot | Description | Example |
|------|-------------|---------|
| **Entities** | Category-level identification | "The image features a women's leisure bicycle" |
| **Attributes** | Visual properties (color, material, texture) | "The primary color is bright purple with glossy finish" |
| **Actions** | Dynamic state / presentation mode | "The bicycle is in fully assembled state" |
| **Relations** | Structural geometry between components | "Handlebars and frame form an upright geometry" |
| **Scene** | Environmental context | "Displayed against a pure white background" |
| **Key Details** | Discriminative local features | "Features a color-matched chain guard" |

Each slot has three constraint types:
- **Required** (entailment): must be present for a correct match
- **Forbidden** (contradiction): indicates a wrong match
- **Ignorable** (invariance): does not affect relevance

Each list contains **0–4** grounded constraints. A slot or list remains empty
when the query provides no corresponding evidence.

## Project Structure

```
EviRank/
├── README.md                          # This file
├── .env.example                       # Empty credential template
├── .gitignore                         # Excludes local and private files
├── requirements.txt                   # Python dependencies
├── configs/
│   └── default.yaml                   # Configuration template
├── src/
│   ├── __init__.py
│   ├── main.py                        # CLI entry point
│   ├── evidence/                      # Evidence mining pipeline
│   │   ├── __init__.py
│   │   ├── evidence_frame.py          # Core data structure (6 slots × 3 labels)
│   │   ├── caption_generator.py       # P-CAP: detailed caption generation
│   │   ├── query_expander.py          # P-B0: query understanding & expansion
│   │   ├── evidence_miner.py          # P-B1: structured evidence slot mining
│   │   └── unified_evidence_generator.py # One-call P-CAP/P-B0/P-B1
│   ├── reranking/                     # Re-ranking pipeline
│   │   ├── __init__.py
│   │   ├── rubric_scorer.py           # Deterministic rubric scoring (Eq. 5)
│   │   └── listwise_ranker.py         # P-RANK: listwise re-ranking (Figure 8)
│   ├── retrieval/                     # Coarse retrieval
│   │   ├── __init__.py
│   │   └── coarse_retrieval.py        # FAISS-based top-K retrieval
│   ├── distillation/                  # Student distillation
│   │   ├── __init__.py
│   │   └── train_student.py           # Training with L_rank + L_slot + L_hard
│   ├── models/                        # Model definitions
│   │   ├── __init__.py
│   │   ├── teacher_model.py           # MLLM teacher (API / open-source)
│   │   ├── student_model.py           # Lightweight student re-ranker
│   │   └── embedding_models.py        # CLIP, EVA-CLIP, BLIP-2 backbones
│   ├── evaluation/                    # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py                 # R@K, MRR, MAP, nDCG
│   ├── generate_evidence.py           # Batch evidence mining (P-CAP+P-B0+P-B1)
│   ├── evaluate_benchmark.py          # Large-scale concurrent evaluation
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Environment-based configuration
│       └── privacy.py                 # Redaction and opaque identifiers
├── scripts/
│   ├── preprocess.sh                  # Build FAISS index
│   ├── run_evirank.sh                 # Run full pipeline
│   ├── evaluate.sh                    # Run evaluation
│   └── train_student.sh               # Train student
└── tests/
    └── test_pipeline_contracts.py     # Pipeline and privacy contract tests
```

Local datasets, indexes, model checkpoints, and generated results are excluded
from version control. Supply credentials through environment variables; never
add real keys to the example configuration or commit local environment files.

## Model Variants

| Variant | Description | Inference Cost |
|---------|-------------|:---:|
| **EviRank-mini** | Rubric scoring only (no MLLM at test time) | Lowest |
| **EviRank** | Distilled student re-ranker | Low |
| **EviRank-plus** | Rubric + MLLM-flash listwise refinement | Medium |
| **EviRank-pro** | Rubric + MLLM-pro listwise refinement | Highest |

- **EviRank-mini**: Uses only the deterministic rubric scorer (Eq. 5) at test time. No MLLM calls needed during inference. Suitable for latency-sensitive applications.
- **EviRank** (distilled): The student model learns from teacher's evidence and ranking signals. At inference, takes only raw query + candidate images (no evidence mining needed).
- **EviRank-plus/pro**: Full teacher pipeline with evidence mining + listwise re-ranking. Differs only in MLLM model capacity.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd EviRank

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy and edit the configuration file:
```bash
cp configs/default.yaml configs/my_config.yaml
```

Key settings to configure:
- `models.teacher_model`: API endpoint and key for MLLM teacher
- `models.embedding_model`: Coarse retrieval backbone
- `dataset`: Dataset paths
- `index`: FAISS index paths
- `reranking.listwise_top_m`: rubric-ranked candidates sent to P-RANK (default 5)

**Environment variables** for sensitive values:
```bash
export TEACHER_API_KEY="your-api-key-here"
export TEACHER_API_URL="https://your-api-endpoint"
export EVIRANK_ID_SALT="generate-a-private-random-value"
```

## Usage

### 1. Preprocess (build FAISS index)

```bash
# Pre-compute embeddings and build FAISS index
python -m src.main --mode preprocess --config configs/default.yaml
```

### 2. Run EviRank Pipeline

```bash
# Text-to-image retrieval with re-ranking
python -m src.main --mode retrieve \
    --config configs/default.yaml \
    --query "A woman in a blue floral dress playing guitar at an outdoor cafe"
```

### 3. Evaluate

```bash
# Evaluate on test split
python -m src.main --mode evaluate \
    --config configs/default.yaml \
    --split test
```

### 4. Train Student Model

```bash
# Distill teacher into student
python -m src.main --mode train \
    --config configs/default.yaml
```

### 5. Generate Evidence (Batch)

```bash
# Pre-compute evidence for a dataset (offline, parallelized)
python -m src.generate_evidence \
    --config configs/default.yaml \
    --input data/sop/test.json \
    --output data/sop/evidence.json \
    --query-type image \
    --workers 32
```

### 6. Large-scale Evaluation (Concurrent)

```bash
# Run EviRank-pro evaluation with 32 concurrent workers
python -m src.evaluate_benchmark \
    --config configs/default.yaml \
    --variant pro \
    --test-path data/sop/test.json \
    --evidence-path data/sop/evidence.json \
    --output results/sop_evirank_pro.json \
    --workers 32
```

## Supported Backbones

| Backbone | Type | Embedding Dim | Usage |
|----------|------|:---:|-------|
| CLIP-ViT-B/32 | Dual-encoder | 512 | Text-to-image |
| CLIP-ViT-L/14 | Dual-encoder | 768 | Text-to-image |
| EVA-CLIP-18B | Dual-encoder | 2048 | Text-to-image |
| BLIP-2 | Dual-encoder | 768 | Text-to-image |
| DINOv2 | Vision-only | 1024 | Image-to-image |

## Benchmarks

EviRank is evaluated on five benchmarks:

| Benchmark | Task | Queries |
|-----------|------|---------|
| Flickr30k | Text → Image | 1,000 test |
| MS-COCO | Text → Image | 5,000 test |
| Stanford Online Products (SoP) | Image → Image | 60,502 test |
| FashionIQ | Composed (Image + Text → Image) | ~6,000 val |
| CIRR | Composed (Image + Text → Image) | ~4,000 test |

## Method Details

### Rubric Scoring (Eq. 5)

The deterministic rubric score for candidate `c` against evidence frame `E(q)`:

```
S_rubric(c) = Σ_{s ∈ slots} w_s · [Σ_{r ∈ required_s} sim(r, desc_c) − Σ_{f ∈ forbidden_s} sim(f, desc_c)]
```

where `sim()` is cosine similarity in embedding space and `w_s` are per-slot weights.

### Distillation Loss (Eq. 10)

```
L_total = confidence · (L_rank + λ · L_slot + η · L_hard)
```

- `L_rank`: KL divergence between student scores and teacher permutation
- `L_slot`: MSE between student scores and per-slot rubric labels
- `L_hard`: Margin loss on teacher-identified hard pairs
- `confidence`: Self-assessed reliability from P-B0

### API Call Contract and Optional Divide-and-Merge

The default teacher pipeline makes exactly two logical API requests per query:
one fused evidence request and one P-RANK request over rubric top-5. Candidate
paths, filenames, directory names, and labels are never included in prompts.

For non-default large listwise candidate sets, divide-and-merge can be enabled
explicitly with `reranking.allow_divide_and_merge: true`:
1. Partition candidates into groups of size B (default B=5)
2. Rank each group independently via teacher
3. Iteratively merge via pairwise re-ranking

This requires `⌈K/B⌉ + ⌈log₂(⌈K/B⌉)⌉` teacher calls.

## Data Preparation

Datasets are **not included** in this repository. To reproduce results:

1. **Flickr30k**: Download from [official source](http://shannon.cs.illinois.edu/DenotationGraph/)
2. **MS-COCO**: Download from [COCO website](https://cocodataset.org/)
3. **Stanford Online Products**: Download from [Stanford](https://cvgl.stanford.edu/projects/lifted_struct/)
4. **FashionIQ**: Download from [FashionIQ repo](https://github.com/XiaoxiaoGuo/fashion-iq)
5. **CIRR**: Download from [CIRR repo](https://github.com/Cuberick-Orion/CIRR)

Place datasets under `data/` following the structure expected by dataset loaders.

## License

This project is released for academic research purposes.
