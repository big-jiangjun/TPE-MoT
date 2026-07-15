<div align="center">

<!-- 标题区域 -->
<h1> TPE-MoT </h1>
<h3> Temporal Perception Expert within Mixture-of-Transformers </h3>
<h4> for Unified BEV and Spatial Reasoning </h4>

<!-- 徽章区域 -->
<p>
  <img src="https://img.shields.io/badge/AAAI-2027-blue?style=flat-square" alt="AAAI-27">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/5B%20Params-lightgrey?style=flat-square" alt="5B">
</p>

<!-- 一句话描述 -->
<p><i>A unified architecture that simultaneously generates explicit BEV representations and answers spatial reasoning questions from historical video sequences.</i></p>

</div>

---

##  Overview

Existing Vision-Language Models (VLMs) for spatial understanding fall into two categories with critical limitations:

- **Pure QA models** (e.g., GeoAlign, 3D-RFT) achieve high VSI-Bench scores but **cannot output explicit 3D spatial representations**.
- **Single-frame BEV models** generate bird's-eye-view maps but **lack language reasoning capabilities**.

**TPE-MoT** bridges this gap by introducing a **Temporal Perception Expert** that operates in parallel with the VLM backbone within a **Mixture-of-Transformers** framework. The key innovation is **layer-wise alignment**: the perception expert and the VLM interact at every transformer layer, enabling fine-grained spatial grounding without task-specific post-training.

<div align="center">

| Capability |  VLM-3R | **TPE-MoT (Ours)** |
|:----------|:------:|:----------------:|
| VSI-Bench Score | 60.9 | **xx** |
| BEV Output | ❌ | **✅** |
| Temporal Video | ❌ | **✅** |
| Parameters| 7B | **5B** |

</div>

> **Key Insight:** We trade a moderate VSI gap for **generality** — a single model that outputs both BEV maps and spatial answers.

---

##  Architecture

### Core Components

```
Input: Historical Video {I_t}_{t=1}^T + 4D Spatial Anchors + Camera Poses
  │
  ├──► ViT Backbone ───────────────────────┐
  │                                          │
  ├──► Pose-Texture Encoder (0.5B) ────────┼──► Layer-wise Cross-Attention ──► VLM (4B)
  │     • Camera Pose Tokens                 │      (at every layer l)
  │     • Dynamic / Static Texture Fusion    │
  │                                          │
  └──► Temporal Perception Expert (0.5B) ────┘
        • Frame-wise: Anchor-to-ViT Cross-Attention
        • Temporal: State Memory / Temporal Attention
        • Output: Spatio-temporal features F_exp^(l)

Output: BEV Feature Map + Language Answer
```

### Layer-wise Alignment (Key Innovation)

Unlike prior MoT approaches (e.g., AutoMoT, UniDriveVLA) that only communicate at a shared latent space, TPE-MoT performs **cross-layer interaction**:

| Layer | VLM Receives from Expert | Expert Receives from VLM |
|:-----:|:------------------------|:-------------------------|
| Shallow (l=1~4) | Low-level geometry & edges | Semantic guidance |
| Middle (l=5~16) | Object-level spatial relations | Contextual priors |
| Deep (l=17~L) | Scene-level BEV semantics | High-level reasoning signals |

---

##  Quick Start

### 1. Installation

```bash
git clone https://anonymous.4open.science/r/tpe-mot-spatial.git
cd tpe-mot-spatial
pip install -r requirements.txt
```

### 2. Evaluation

**VSI-Bench Spatial Reasoning:**
```bash
bash scripts/eval_vsi.sh   --config configs/evaluation/vsi_bench.yaml
```

**nuScenes BEV Reconstruction:**
```bash
bash scripts/eval_bev.sh   --config configs/evaluation/bev_nuscenes.yaml
```

---

##  Results

### VSI-Bench (Video Spatial Intelligence)

<div align="center">

| Model | Params | Avg | BEV | Temporal |
|:-----:|:------:|:---:|:---:|:--------:|
| GPT-4o | — | 34.0 | ❌ | ❌ |
| Gemini-2.5-Pro | — | 53.6 | ❌ | ❌ |
| InternVL3-78B | 78B | 48.4 | ❌ | ❌ |
| Cambrian-S-3B | 3B | 57.3 | ❌ | ❌ |
| VLM-3R-7B | 7B | 60.9 | ❌ | ❌ |
| **GeoAlign-4B** | **4B** | **71.4** | ❌ | ❌ |
| **TPE-MoT (Ours)** | **5B** | **62.0** | **✅** | **✅** |

</div>

> **Note:** GeoAlign and 3D-RFT are specialized post-training methods limited to spatial QA. TPE-MoT is the **only model** in this family that simultaneously generates explicit BEV representations.

### BEV Reconstruction (nuScenes)

<div align="center">

| Method | mIoU ↑ | HD ↓ | Notes |
|:-------|:------:|:----:|:------|
| BEVFormer | 56.9 | 0.82 | Single-task BEV only |
| BEVFusion | 58.2 | 0.78 | Requires LiDAR |
| PETR | 55.4 | 0.85 | Single-task BEV only |
| **TPE-MoT (Ours)** | **XX.X** | **XX.X** | **Joint BEV + VSI** |

</div>

### Ablation Studies

| Configuration | VSI-Bench | BEV mIoU | Analysis |
|:-------------|:---------:|:--------:|:---------|
| Full Model | 62.0 | XX.X | Baseline |
| w/o Perception Expert | ↓ | ↓ | Expert is essential for both tasks |
| w/o Pose Tokens | ↓ | ↓ | Camera pose provides geometric prior |
| w/o Texture Encoder | ↓ | ↓ | Dynamic texture aids moving objects |
| Top-layer Alignment Only | ↓ | ↓ | Layer-wise alignment is critical |
| Single-frame (no temporal) | ↓ | ↓ | Temporal modeling improves both tasks |
| 3D Anchors (no time dim) | ↓ | ↓ | 4D anchors capture dynamics |

---

##  Repository Structure

```
tpe-mot-spatial/
├── configs/              # Model & evaluation configurations
├── models/               # Core architecture (open-sourced for review)
│   ├── tpe_mot.py
│   ├── temporal_perception_expert.py
│   ├── layerwise_alignment.py      ⭐ Key Innovation
│   ├── pose_texture_encoder.py
│   └── bev_head.py
├── evaluation/           # Evaluation scripts & metrics
├── data_preprocess/      # Data loaders (nuScenes, VSI-298K)
├── scripts/              # One-click evaluation
├── supplementary/        # Pseudocode & implementation details
│   ├── pseudocode/
│   ├── architecture_details.md
│   └── hyperparameters.md
└── tests/                # Unit tests for core modules
```

For detailed architecture descriptions, see [`supplementary/architecture_details.md`](supplementary/architecture_details.md).

---

##  Code Availability

Due to intellectual property considerations during the review process, we provide an **anonymous repository** containing the **core architecture implementations** (perception expert, layer-wise alignment module, and BEV head) and **evaluation scripts** to facilitate reproducibility review.

The **complete training pipeline and pre-trained weights** will be released upon paper acceptance.

---

##  Citation

```bibtex
@inproceedings{anonymous2027tpemot,
  title={Temporal Perception Expert within Mixture-of-Transformers for Unified BEV and Spatial Reasoning},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2027}
}
```

> **Note:** Author information will be updated upon paper acceptance.

---

<div align="center">

**Built with PyTorch** · **Anonymous Submission for AAAI-27**

</div>
